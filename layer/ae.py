import numpy as np
import einops
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as thf
import pytorch_lightning as pl
import torchvision
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, \
    AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.model import Encoder
import lpips
from kornia import color


def normalize_per_image(x, eps=1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def sobel_edge_map(x):
    if x.shape[1] == 3:
        gray = (
                0.2989 * x[:, 0:1] +
                0.5870 * x[:, 1:2] +
                0.1140 * x[:, 2:3]
        )
    else:
        gray = x

    sobel_x = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1., -2., -1.],
         [0., 0., 0.],
         [1., 2., 1.]],
        device=x.device,
        dtype=x.dtype
    ).view(1, 1, 3, 3)

    gx = thf.conv2d(gray, sobel_x, padding=1)
    gy = thf.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    return edge


def one_channel_to_three(x):
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def high_frequency_map(x, kernel_size=7):
    if x.shape[1] == 3:
        gray = (
                0.2989 * x[:, 0:1] +
                0.5870 * x[:, 1:2] +
                0.1140 * x[:, 2:3]
        )
    else:
        gray = x
    blur = thf.avg_pool2d(gray, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    hf = torch.abs(gray - blur)
    return normalize_per_image(hf)


def map01_to_log_image(x):
    x = one_channel_to_three(x)
    return x * 2.0 - 1.0


def psnr_metric(pred, target, eps=1e-8):
    pred01 = (pred.clamp(-1, 1) + 1.0) / 2.0
    target01 = (target.clamp(-1, 1) + 1.0) / 2.0
    mse = torch.mean((pred01 - target01) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean(), mse.mean()


def linear_ramp_weight(global_step, start_step, ramp_steps, max_weight):
    if start_step is None:
        return 0.0
    step = float(global_step)
    start = float(start_step)
    if step < start:
        return 0.0
    if ramp_steps <= 0:
        return float(max_weight)
    t = (step - start) / float(ramp_steps)
    t = max(0.0, min(1.0, t))
    return float(max_weight) * t


def edge_texture_residual_constraint(
        x_w,
        x_rec,
        alpha_res=2.0,
        edge_weight=0.5,
        texture_weight=0.5,
        texture_kernel_size=7,
        base_penalty=0.2,
        max_penalty=5.0,
):
    x_rec_ref = x_rec.detach()
    r_wm = x_w - x_rec_ref

    # 1. Edge map
    edge = normalize_per_image(sobel_edge_map(x_rec_ref))

    # 2. Texture / high-frequency map
    texture = high_frequency_map(x_rec_ref, kernel_size=texture_kernel_size)

    # 3. Joint edge-texture map
    denom = max(edge_weight + texture_weight, 1e-8)
    e_joint = (edge_weight * edge + texture_weight * texture) / denom
    e_joint = e_joint.clamp(0.0, 1.0)

    # 4. Mask: flat region high, edge/texture region low
    mask = 1.0 - e_joint

    # 5. Safe exponential penalty
    w_res = base_penalty + mask * torch.exp(alpha_res * mask)
    w_res = torch.clamp(w_res, min=base_penalty, max=max_penalty)

    # 6. Weighted residual loss
    loss_res = ((w_res * r_wm) ** 2).mean()

    return loss_res, w_res, r_wm, e_joint, edge, texture


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SecretEncoder3(nn.Module):
    def __init__(self, secret_len, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * 3),
            nn.SiLU(),
            View(-1, 3, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 3, 3, 3, padding=1))
        )  # secret len -> ch x res x res

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None


class SecretEncoder4(nn.Module):
    """same as SecretEncoder3 but with ch as input"""

    def __init__(self, secret_len, ch=3, base_res=16, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ch, ch, 3, padding=1))
        )  # secret len -> ch x res x res

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        return c, None


class SecretEncoder6(nn.Module):
    """join img emb with secret emb"""

    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, emode='c3') -> None:
        super().__init__()
        assert emode in ['c3', 'c2', 'm3']

        if emode == 'c3':  # c3: concat c and x each has ch channels
            secret_ch = ch
            join_ch = 2 * ch
        elif emode == 'c2':  # c2: concat c (2) and x ave (1)
            secret_ch = 2
            join_ch = ch
        elif emode == 'm3':  # m3: multiply c (ch) and x (ch)
            secret_ch = ch
            join_ch = ch

            # m3: multiply c (ch) and x ave (1)
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.emode = emode
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * secret_ch),
            nn.SiLU(),
            View(-1, secret_ch, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        self.join_encoder = nn.Sequential(
            conv_nd(2, join_ch, join_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, join_ch, ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, ch, ch, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.emode == 'c3':
            x = torch.cat([x, c], dim=1)
        elif self.emode == 'c2':
            x = torch.cat([x.mean(dim=1, keepdim=True), c], dim=1)
        elif self.emode == 'm3':
            x = x * c
        dx = self.join_encoder(x)
        dx = self.out_layer(dx)
        return dx, None


class SecretEncoder5(nn.Module):
    """same as SecretEncoder3 but with ch as input"""

    def __init__(self, secret_len, ch=3, base_res=16, resolution=64, joint=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.joint = joint
        self.resolution = resolution
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base), 2 ** (log_resolution - log_base))),
            # chx16x16 -> chx256x256
        )  # secret len -> ch x res x res
        if joint:
            self.join_encoder = nn.Sequential(
                conv_nd(2, 2 * ch, 2 * ch, 3, padding=1),
                nn.SiLU(),
                conv_nd(2, 2 * ch, ch, 3, padding=1),
                nn.SiLU()
            )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        if self.joint:
            x = thf.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False,
                                antialias=True)
            c = self.join_encoder(torch.cat([x, c], dim=1))
        c = self.out_layer(c)
        return c, None


class SecretEncoder2(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.encoder.conv_out = zero_module(self.encoder.conv_out)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32 * 32 * ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2 ** (log_resolution - 5), 2 ** (log_resolution - 5))),  # chx16x16 -> chx256x256
            # zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        # self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        posterior = h
        return posterior

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = torch.cat([x, c], dim=1)
        z = self.encode(x)
        # z = self.out_layer(z)
        return z, None


class SecretEncoder(nn.Module):
    def __init__(self, secret_len, embed_dim, ddconfig, ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False) -> None:
        super().__init__()
        log_resolution = int(np.log2(ddconfig.resolution))
        self.secret_len = secret_len
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, 32 * 32 * ddconfig.out_ch),
            nn.SiLU(),
            View(-1, ddconfig.out_ch, 32, 32),
            nn.Upsample(scale_factor=(2 ** (log_resolution - 5), 2 ** (log_resolution - 5))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))
        )  # secret len -> ch x res x res
        # out_resolution = ddconfig.resolution//(len(ddconfig.ch_mult)-1)
        self.out_layer = zero_module(conv_nd(2, ddconfig.out_ch, ddconfig.out_ch, 3, padding=1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        misses, ignores = self.load_state_dict(sd, strict=False)
        print(f"[SecretEncoder] Restored from {path}, misses: {misses}, ignores: {ignores}")

    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        self.encoder.load_state_dict(ae_model.encoder.state_dict())
        self.quant_conv.load_state_dict(ae_model.quant_conv.state_dict())

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.secret_scaler(c)
        x = x + c
        posterior = self.encode(x)
        z = posterior.sample()
        z = self.out_layer(z)
        return z, posterior



class CurvedMaskSecretEncoder(nn.Module):
    def __init__(self, secret_len, ch=3, base_res=32, resolution=64, hidden_ch=32, emode='concat') -> None:
        super().__init__()
        assert emode in ['concat', 'mul']
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.resolution = resolution
        self.emode = emode
        self.ch = ch

        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res * base_res * ch),
            nn.SiLU(),
            View(-1, ch, base_res, base_res),
            nn.Upsample(scale_factor=(2 ** (log_resolution - log_base),
                                      2 ** (log_resolution - log_base))),
        )

        in_ch = ch * 2 if emode == 'concat' else ch * 3
        self.join_encoder = nn.Sequential(
            conv_nd(2, in_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, hidden_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, hidden_ch, ch, 3, padding=1),
            nn.SiLU(),
        )
        self.out_layer = zero_module(conv_nd(2, ch, ch, 3, padding=1))

    def copy_encoder_weight(self, ae_model):
        return None

    def encode_secret(self, c):
        return self.secret_scaler(c)

    def forward(self, x, c):
        # x: image latent [B, 3, H, W]
        # c: secret [B, secret_len]
        s_map = self.encode_secret(c)
        if s_map.shape[-2:] != x.shape[-2:]:
            s_map = thf.interpolate(s_map, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.emode == 'concat':
            h = torch.cat([x, s_map], dim=1)
        else:
            h = torch.cat([x, s_map, x * s_map], dim=1)

        h = self.join_encoder(h)
        eps = self.out_layer(h)
        return eps, None



class CurvedMaskLatentAdapter(nn.Module):
    def __init__(self, latent_ch=3, hidden_ch=32, output_scale=0.05):
        super().__init__()
        self.output_scale = float(output_scale)
        self.net = nn.Sequential(
            conv_nd(2, latent_ch * 2, hidden_ch, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, hidden_ch, hidden_ch, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(2, hidden_ch, latent_ch, 3, padding=1)),
        )

    def forward(self, z, frozen_secret_eps):
        adapter_input = torch.cat([z, frozen_secret_eps], dim=1)
        return self.output_scale * self.net(adapter_input)



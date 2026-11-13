import math
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia
import kornia.augmentation as K_aug


class CurvedPhysicalS2CLayer(nn.Module):

    def _apply_contrast(self, x):
        scale = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(*self.contrast_range)
        return torch.clamp(x * scale, 0, 1)

    def _apply_saturation(self, x):
        sat = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(*self.sat_range)
        weight = torch.tensor([0.3, 0.6, 0.1], device=x.device).view(1, 3, 1, 1)
        gray = torch.sum(x * weight, dim=1, keepdim=True)
        return torch.clamp((1 - sat) * x + sat * gray, 0, 1)

    def _apply_eye_protection(self, x):
        B = x.size(0)
        sev = torch.empty((B, 1, 1, 1), device=x.device).uniform_(*self.eye_severity_range)
        r_gain = 1.05 + 0.05 * sev
        g_gain = 0.90 + 0.10 * sev
        b_gain = 0.80 - 0.10 * sev
        mult = torch.cat([r_gain, g_gain, b_gain], dim=1)
        x_mult = torch.clamp(x * mult, 0, 1)
        gamma = 1.0 - 0.05 * sev
        return torch.clamp(torch.pow(x_mult + 1e-8, gamma), 0, 1)

    def _apply_light_distortion(self, x):
        B, C, H, W = x.shape
        device = x.device
        edge_darkness = torch.empty(B, device=device).uniform_(*self.edge_darkness_range)
        x_c = torch.randint(int(H * 0.3), int(H * 0.7) + 1, (B,), device=device)
        y_c = torch.randint(int(W * 0.3), int(W * 0.7) + 1, (B,), device=device)
        y_grid = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)
        x_grid = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
        dist = torch.sqrt((y_grid - x_c.view(B, 1, 1)) ** 2 + (x_grid - y_c.view(B, 1, 1)) ** 2)
        max_dist = dist.amax(dim=(1, 2), keepdim=True) + 1e-5
        m = 1.0 - (dist / max_dist) * (1.0 - edge_darkness.view(B, 1, 1))
        mask = m.unsqueeze(1).expand(-1, C, -1, -1)
        return torch.clamp(x * mask, 0, 1)


    def _apply_sensor_noise(self, x):
        B = x.size(0)
        device = x.device
        poisson_scale = torch.empty((B, 1, 1, 1), device=device).uniform_(*self.poisson_scale_range)
        gauss_std = self.noise_std
        noise_std_map = torch.sqrt(torch.clamp(x, min=0.0) * poisson_scale + gauss_std ** 2)
        noise = torch.randn_like(x) * noise_std_map
        return torch.clamp(x + noise, 0, 1)

    def _apply_jpeg(self, x):
        try:
            jpeg_layer = K_aug.RandomJPEG(jpeg_quality=self.jpeg_quality_range, p=1.0).to(x.device)
            return jpeg_layer(x)
        except Exception:
            return x



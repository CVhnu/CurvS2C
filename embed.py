
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from ldm.util import instantiate_from_config
from tools.helpers import welcome_message
from tools.ecc import ECC


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_secret_bits(path, secret_len):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace(",", " ").strip()

    bits = [int(x) for x in text.split()]
    if len(bits) != secret_len:
        raise ValueError(
            f"Expected {secret_len} bits, got {len(bits)}"
        )
    if any(x not in (0, 1) for x in bits):
        raise ValueError("Secret bits must be 0 or 1")

    return np.asarray(bits, dtype=np.float32)


def main(args):
    print(welcome_message())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # 1. 加载模型
    config = OmegaConf.load(args.config).model
    secret_len = int(
        config.params.control_config.params.secret_len
    )
    config.params.decoder_config.params.secret_len = secret_len

    model = instantiate_from_config(config)

    checkpoint = torch.load(
        args.weight, map_location="cpu"
    )

    if "global_step" in checkpoint:
        print(
            f"Global step: {checkpoint['global_step']}, "
            f"epoch: {checkpoint.get('epoch', 'unknown')}"
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    misses, ignores = model.load_state_dict(
        state_dict, strict=False
    )

    print(f"Missed keys: {misses}")
    print(f"Ignored keys: {ignores}")

    model = model.to(device).eval()

    # 2. 图像预处理
    tform = transforms.Compose([
        transforms.Resize(
            (args.image_size, args.image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    # 3. 加载水印信息
    if args.secret_bits is not None:
        secret_np = load_secret_bits(
            args.secret_bits, secret_len
        )[None, :]

        print(
            f"Using fixed secret bits from: "
            f"{args.secret_bits}"
        )
    else:
        ecc = ECC()
        secret_np = ecc.encode_text([args.secret])
        print(f"Using ECC text secret: {args.secret}")

    secret = torch.from_numpy(
        np.asarray(secret_np, dtype=np.float32)
    ).to(device)

    # 4. 读取输入图像
    cover_dir = Path(args.cover_dir).expanduser()

    if not cover_dir.is_dir():
        raise FileNotFoundError(
            f"Cover directory not found: {cover_dir}"
        )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in cover_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No valid images found in: {cover_dir}"
        )

    # 5. 嵌入水印并保存
    with torch.no_grad():
        for idx, img_path in enumerate(
            image_paths, start=1
        ):
            print(
                f"[{idx}/{len(image_paths)}] "
                f"Embedding: {img_path.name}"
            )

            with Image.open(img_path) as im:
                cover_org = im.convert("RGB")

            w, h = cover_org.size

            cover_np = np.asarray(
                cover_org, dtype=np.uint8
            )

            cover = tform(
                cover_org
            ).unsqueeze(0).to(device)

            z = model.encode_first_stage(cover)

            z_embed, _ = model(z, None, secret)

            stego = model.decode_first_stage(z_embed)

            residual = stego.clamp(-1, 1) - cover

            residual = F.interpolate(
                residual,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )

            residual = (
                residual
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()[0]
            )

            stego_uint8 = np.clip(
                residual + cover_np / 127.5 - 1.0,
                -1.0,
                1.0
            ) * 127.5 + 127.5

            stego_uint8 = stego_uint8.astype(
                np.uint8
            )

            # 保存水印图像
            output_path = (
                output_dir / f"{img_path.stem}.png"
            )

            Image.fromarray(
                stego_uint8
            ).save(output_path)

            print(f"Saved: {output_path}")

    print(
        f"\nDone: embedded {len(image_paths)} image(s)."
    )
    print(
        f"Output folder: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed watermark only"
    )

    parser.add_argument(
        "-c", "--config",
        default="models/VQ4_mir_inference.yaml"
    )

    parser.add_argument(
        "-w", "--weight",
        required=True
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--secret",
        default="secrets"
    )

    parser.add_argument(
        "--secret_bits",
        default="/secret_100_bits.txt"
    )

    parser.add_argument(
        "--cover_dir",
        required=True
    )

    parser.add_argument(
        "-o", "--output_dir",
        required=True
    )

    args = parser.parse_args()

    if args.secret_bits == "None":
        args.secret_bits = None

    main(args)

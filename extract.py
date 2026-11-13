
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ldm.util import instantiate_from_config
from tools.helpers import welcome_message


DEFAULT_CONFIG = "models/VQ4_mir_inference.yaml"
DEFAULT_WEIGHT = ""
DEFAULT_SECRET_BITS = ""
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_secret_bits(path, secret_len):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().replace(",", " ").strip()

    bits = [int(value) for value in text.split()]
    if len(bits) != secret_len:
        raise ValueError("Expected {} bits, got {}".format(secret_len, len(bits)))
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Secret bits must contain only 0 and 1")
    return np.asarray(bits, dtype=np.int32)


def list_images(images_dir):
    folder = Path(images_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError("Image folder does not exist: {}".format(folder))
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_model(config_path, weight_path, device):
    config = OmegaConf.load(config_path).model
    secret_len = int(config.params.control_config.params.secret_len)
    config.params.decoder_config.params.secret_len = secret_len
    model = instantiate_from_config(config)

    checkpoint = torch.load(weight_path, map_location="cpu")
    if "global_step" in checkpoint:
        print(
            "Global step: {}, epoch: {}".format(
                checkpoint["global_step"], checkpoint.get("epoch", "unknown")
            )
        )
    state_dict = checkpoint.get("state_dict", checkpoint)
    misses, ignores = model.load_state_dict(state_dict, strict=False)
    print("Missed keys: {}\nIgnored keys: {}".format(misses, ignores))

    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, secret_len


def parse_args():
    parser = argparse.ArgumentParser(description="Decode images using the encoded fixed bit file")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", help="Single image to decode")
    inputs.add_argument("--stego_dir", help="Folder containing images to decode")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("-w", "--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--secret_bits", default=DEFAULT_SECRET_BITS)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument(
        "--eval_bits",
        type=int,
        default=None,
        help="Number of leading bits to evaluate (default: all model bits)",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Per-image result CSV (default: decode_results.csv in the input folder)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(welcome_message())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model, secret_len = load_model(args.config, args.weight, device)
    expected_bits = load_secret_bits(args.secret_bits, secret_len)

    eval_bits = secret_len if args.eval_bits is None else args.eval_bits
    if eval_bits < 1 or eval_bits > secret_len:
        raise ValueError("--eval_bits must be between 1 and {}".format(secret_len))
    expected_eval = expected_bits[:eval_bits]

    if args.image:
        image_paths = [Path(args.image).expanduser()]
        default_csv_dir = image_paths[0].resolve().parent
    else:
        image_paths = list_images(args.stego_dir)
        default_csv_dir = Path(args.stego_dir).expanduser().resolve()
    if not image_paths:
        raise RuntimeError("No supported images were found")

    output_csv = Path(args.output_csv).expanduser() if args.output_csv else default_csv_dir / "decode_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("Secret bits:", args.secret_bits)
    print("Model payload: {} bits".format(secret_len))
    print("Evaluated payload: first {} bits".format(eval_bits))

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        try:
            with Image.open(image_path) as image:
                image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model.decoder(image_tensor)
                predicted_bits = (logits > 0).to(torch.int32).cpu().numpy().reshape(-1)

            if predicted_bits.size < secret_len:
                raise ValueError(
                    "Decoder returned {} bits; expected at least {}".format(predicted_bits.size, secret_len)
                )

            predicted_eval = predicted_bits[:eval_bits]
            bit_errors = int(np.count_nonzero(predicted_eval != expected_eval))
            bit_accuracy = float(1.0 - bit_errors / float(eval_bits))
            message_success = bit_errors == 0

            rows.append(
                {
                    "filename": image_path.name,
                    "bit_accuracy": bit_accuracy,
                    "bit_errors": bit_errors,
                    "message_success": int(message_success),
                    "evaluated_bits": eval_bits,
                    "predicted_bits": "".join(str(bit) for bit in predicted_bits[:secret_len]),
                    "status": "ok",
                }
            )
            print(
                "[{}/{}] {} | bit acc: {:.4f} | errors: {} | message: {}".format(
                    index,
                    len(image_paths),
                    image_path.name,
                    bit_accuracy,
                    bit_errors,
                    "success" if message_success else "failed",
                )
            )
        except Exception as error:
            rows.append(
                {
                    "filename": image_path.name,
                    "bit_accuracy": "",
                    "bit_errors": "",
                    "message_success": 0,
                    "evaluated_bits": eval_bits,
                    "predicted_bits": "",
                    "status": "failed: {}".format(error),
                }
            )
            print("[{}/{}] {} | failed: {}".format(index, len(image_paths), image_path.name, error))

    fieldnames = [
        "filename",
        "bit_accuracy",
        "bit_errors",
        "message_success",
        "evaluated_bits",
        "predicted_bits",
        "status",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["status"] == "ok"]
    if not valid_rows:
        raise RuntimeError("All images failed to decode; see {}".format(output_csv))

    accuracies = np.asarray([row["bit_accuracy"] for row in valid_rows], dtype=np.float64)
    total_errors = int(sum(row["bit_errors"] for row in valid_rows))
    total_bits = len(valid_rows) * eval_bits
    message_successes = int(sum(row["message_success"] for row in valid_rows))

    print("\n================ Decode Summary =================")
    print("Decoded images      : {}/{}".format(len(valid_rows), len(rows)))
    print("Evaluated bits/image: {}".format(eval_bits))
    print("Average bit accuracy: {:.6f} +/- {:.6f}".format(float(accuracies.mean()), float(accuracies.std())))
    print("Aggregate bit acc   : {:.6f} ({}/{})".format(1.0 - total_errors / float(total_bits), total_bits - total_errors, total_bits))
    print(
        "Message success rate: {:.6f} ({}/{})".format(
            message_successes / float(len(valid_rows)), message_successes, len(valid_rows)
        )
    )
    print("CSV saved to        : {}".format(output_csv.resolve()))
    print("===========================================================")


if __name__ == "__main__":
    main()

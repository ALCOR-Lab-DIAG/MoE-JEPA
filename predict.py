"""
Classify images as real / fake (fully synthetic) / tampered with MoE-JEPA.

Usage:
    python predict.py --checkpoint moejepa_sidset.safetensors image1.jpg image2.png
"""
import argparse
import torch
from PIL import Image

from model.model import load_moejepa, load_processor, preprocess_image, BACKBONE_NAME, LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-JEPA single-image inference")
    parser.add_argument("images", nargs="+", help="Image files to classify")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Released .safetensors weights or a .ckpt produced by train.py")
    parser.add_argument("--model_name", type=str, default=BACKBONE_NAME,
                        help=f"V-JEPA 2 backbone, HF id or local path (default: {BACKBONE_NAME})")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model = load_moejepa(args.checkpoint, model_name=args.model_name, device=args.device)
    processor = load_processor(args.model_name)

    for path in args.images:
        x = preprocess_image(Image.open(path), processor).unsqueeze(0).to(args.device)
        probs = model(x).float().softmax(dim=-1)[0]
        scores = "  ".join(f"{name}={p:.3f}" for name, p in zip(LABELS, probs.tolist()))
        print(f"{path}: {LABELS[probs.argmax()]}  ({scores})")


if __name__ == "__main__":
    main()

"""
Convert a Lightning checkpoint from train/train.py into the release format:
a .safetensors file holding only the trained weights (MoE routers and routed
experts, MIL pooler, noise stream, fusion gate, classifier) plus the
hyperparameters as metadata. The frozen V-JEPA 2 backbone and the frozen shared
experts are left out; they are rebuilt from the Hugging Face Hub at load time.

Usage:
    python scripts/export_checkpoint.py --ckpt path/to/best.ckpt --out moejepa_sidset.safetensors
"""
import json
import argparse
import torch
from safetensors.torch import save_file

TRAINED_BACKBONE_KEYS = (".mlp.router.", ".mlp.routed_experts.")


def is_trained(key):
    return not key.startswith("model.") or any(t in key for t in TRAINED_BACKBONE_KEYS)


def main():
    parser = argparse.ArgumentParser(description="Export MoE-JEPA trained weights")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning .ckpt file")
    parser.add_argument("--out", type=str, required=True, help="Output .safetensors file")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    if hparams.get("unfreeze_last_n_blocks", 0) > 0:
        raise ValueError("Backbone blocks were fine-tuned; exporting only the heads would lose them.")
    hparams.pop("model_name", None)

    state_dict = {k: v.contiguous() for k, v in ckpt["state_dict"].items() if is_trained(k)}
    metadata = {
        "hyper_parameters": json.dumps(hparams),
        "epoch": str(ckpt.get("epoch")),
        "global_step": str(ckpt.get("global_step")),
    }
    save_file(state_dict, args.out, metadata=metadata)

    n_params = sum(v.numel() for v in state_dict.values())
    print(f"Saved {len(state_dict)} tensors ({n_params / 1e6:.1f}M params) to {args.out}")


if __name__ == "__main__":
    main()

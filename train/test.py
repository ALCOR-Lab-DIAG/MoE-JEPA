import os
import sys
import argparse
import torch
import lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.model import load_moejepa, load_processor, preprocess_image, BACKBONE_NAME, NUM_FRAMES


class PrepareTransform:
    """Picklable transform wrapper (lambdas can't be pickled with forkserver)."""
    def __init__(self, processor, num_frames):
        self.processor = processor
        self.num_frames = num_frames

    def __call__(self, examples):
        pixel_values = [preprocess_image(img, self.processor, self.num_frames) for img in examples['image']]
        return {
            'pixel_values': torch.stack(pixel_values),
            'label': torch.tensor(examples['label'])
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Test MoE-JEPA Deepfake Detector")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Released .safetensors weights or a .ckpt produced by train.py"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Local dataset saved with save_to_disk(), or a Hugging Face Hub dataset id"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Split to evaluate if the dataset has several (default: test)"
    )
    parser.add_argument(
        "--model_name", type=str, default=BACKBONE_NAME,
        help=f"V-JEPA 2 backbone, HF id or local path (default: {BACKBONE_NAME})"
    )
    parser.add_argument(
        "--num_frames", type=int, default=NUM_FRAMES,
        help=f"Frames each image is repeated to (default: {NUM_FRAMES}, as in training)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for testing (default: 16)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of dataloader workers (default: 4)"
    )
    return parser.parse_args()


def load_split(path, split):
    dataset = load_from_disk(path) if os.path.exists(path) else load_dataset(path)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            if len(dataset) != 1:
                raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
            split = next(iter(dataset))
        dataset = dataset[split]
    return dataset


def test(args):
    model = load_moejepa(args.checkpoint, model_name=args.model_name)
    processor = load_processor(args.model_name)

    test_dataset = load_split(args.dataset, args.split)
    if args.limit is not None:
        test_dataset = test_dataset.select(range(min(args.limit, len(test_dataset))))
    print(test_dataset)

    print(f"Testing the model: {args.checkpoint}")
    print(f"Running test on dataset: {args.dataset}")
    print(f"Backbone: {args.model_name}")

    test_dataset.set_transform(PrepareTransform(processor, args.num_frames))
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0
    )

    cuda = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if cuda else "cpu",
        devices=1,
        logger=False,
        precision="bf16-mixed" if cuda else "32-true",
    )

    results = trainer.test(model, test_dataloader)

    log_file_path = "test_results_json.txt"
    with open(log_file_path, "a") as f:
        f.write("--- Deepfake Detection Test Results ---\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Backbone: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset} [{args.split}]\n")
        f.write("-" * 40 + "\n")

        for i, res in enumerate(results):
            f.write(f"Results for Dataloader {i}:\n")
            for metric, value in res.items():
                f.write(f"{metric}: {value:.4f}\n")

        f.write("-" * 40 + "\n")
        f.write("*" * 80 + "\n")
        f.write("*" * 80 + "\n\n\n")

    print(f"Results successfully saved to {log_file_path}")
    return

if __name__ == "__main__":
    args = parse_args()
    test(args)

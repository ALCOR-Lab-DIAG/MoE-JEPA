import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoVideoProcessor, AutoImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as pl
import sys
import argparse
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy
import os
import json 
from datasets import load_from_disk, load_dataset

sys.path.append("..")
from model.model import MoEJEPADeepfakeDetector

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def prepare_sidset(examples, processor):
    processed_images = []
    for img in examples['image']:
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")

            inputs = processor(img, return_tensors="pt")

            # Auto-detect: video processors return 'pixel_values_videos',
            # image processors return 'pixel_values'
            if 'pixel_values_videos' in inputs:
                pixel_values = inputs['pixel_values_videos']
                pixel_values = pixel_values.repeat(1, 2, 1, 1, 1)
                pixel_values = pixel_values.squeeze(0)
            else:
                pixel_values = inputs['pixel_values']
                pixel_values = pixel_values.squeeze(0)  # [1, 3, H, W] -> [3, H, W]

            processed_images.append(pixel_values)
        except Exception as e:
            print(f"Skipping corrupted image. Error: {e}")
            continue

    pixel_values_tot = torch.stack(processed_images)

    return {
        'pixel_values': pixel_values_tot,
        'label': torch.tensor(examples['label'])
    }

class PrepareTransform:
    """Picklable transform wrapper (lambdas can't be pickled with forkserver)."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        return prepare_sidset(examples, self.processor)

def is_corrupt(example):
    try:
        img = example['image']
        if hasattr(img, 'getexif'):
            img.getexif() 
        return False
    except Exception:
        return True

PIXEL_LIMIT = 89478485 

def is_safe_size(example):
    img = example['image']
    width, height = img.size 
    return (width * height) <= PIXEL_LIMIT

def filter_sidset(dataset):
    # labels_train = np.array(dataset['test']['label'])
    # keep_indices = np.where(labels_train != 2)[0]
    # dataset['test'] = dataset['test'].select(keep_indices)
    # labels_val = np.array(dataset['validation']['label'])
    # keep_indices = np.where(labels_val != 2)[0]
    # dataset['validation'] = dataset['validation'].select(keep_indices)
    labels_train = np.array(dataset['label'])
    keep_indices = np.where(labels_train != 2)[0]
    dataset= dataset.select(keep_indices)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Test MoE-JEPA Deepfake Detector")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .ckpt checkpoint file"
    )
    parser.add_argument(
        "--dataset", type=str,
        default="<<YOUR LOCAL PATH>>",
        help="Path to the test dataset"
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

def test(args):
    MODEL_NAME = args.model_name
    checkpoint_path = args.checkpoint
    dataset_path = args.dataset

    model = MoEJEPADeepfakeDetector.load_from_checkpoint(
        checkpoint_path, model_name=MODEL_NAME, strict=False, map_location="cuda"
    )
    model.to("cuda")
    model.eval()
    
    try:
        processor = AutoVideoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Loaded VideoProcessor for: {MODEL_NAME}")
    except (ValueError, KeyError):
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print(f"Loaded ImageProcessor for: {MODEL_NAME}")

    # Load Dataset
    test_dataset = load_from_disk(dataset_path)

    print(test_dataset)

    # print("filtering tampered images...")
    # test_dataset = filter_sidset(test_dataset)

    if 'train' in test_dataset:
        test_dataset['test'] = test_dataset.pop('train')

    if isinstance(test_dataset, dict) or hasattr(test_dataset, 'keys'):
        if 'test' in test_dataset:
            test_dataset = test_dataset['test']

    print(f"Testing the model: {checkpoint_path}")
    print(f"Running test on dataset: {dataset_path}")
    print(f"Backbone: {MODEL_NAME}")

    test_dataset.set_transform(PrepareTransform(processor))
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=True
    )

    custom_ddp_strategy = DDPStrategy(
        timeout=timedelta(seconds=7200),  # 2 hours: protects against large epoch-end syncs
        find_unused_parameters=True,      # Required for MoE if some experts aren't used in a batch
        gradient_as_bucket_view=True      # Reduces memory fragmentation on 40GB cards
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, 
        logger=False,
        strategy=custom_ddp_strategy,
        precision="bf16-mixed", 
    )

    results = trainer.test(model, test_dataloader)


    log_file_path = "test_results_json.txt"
    with open(log_file_path, "a") as f:
        f.write("--- Deepfake Detection Test Results ---\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Backbone: {MODEL_NAME}\n")
        f.write(f"Dataset: {dataset_path}\n")
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
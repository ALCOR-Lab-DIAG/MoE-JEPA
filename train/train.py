import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoVideoProcessor, AutoImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as pl
import sys
import os
import wandb
from datetime import timedelta
from datasets import load_dataset, load_from_disk
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import transforms
from lightning.pytorch.strategies import DDPStrategy
from types import SimpleNamespace
from functools import partial

SEED = 9999

sys.path.append("..")
from model.model import MoEJEPADeepfakeDetector3ClassesMOD

## Optional Wandb login
#wandb_key = "<<YOUR KEY>>"
#wandb.login(key=wandb_key)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def filter_sidset(dataset):
    labels_train = np.array(dataset['train']['label'])
    keep_indices = np.where(labels_train != 2)[0]
    dataset['train'] = dataset['train'].select(keep_indices)
    labels_val = np.array(dataset['validation']['label'])
    keep_indices = np.where(labels_val != 2)[0]
    dataset['validation'] = dataset['validation'].select(keep_indices)
    return dataset

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

def prepare_sidset(examples, processor=None):
    if processor is None:
        raise ValueError("Processor is required")

    processed_images = []
    
    # Process each example in the batch
    for img, label in zip(examples['image'], examples['label']):
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img = augment_transform(img)

        inputs = processor(img, return_tensors="pt")
        if 'pixel_values_videos' in inputs:
            pixel_values = inputs['pixel_values_videos']
            pixel_values = pixel_values.repeat(1, 4, 1, 1, 1)
            pixel_values = pixel_values.squeeze(0)
        else:
            pixel_values = inputs['pixel_values']
            pixel_values = pixel_values.squeeze(0)
            
        processed_images.append(pixel_values)

    pixel_values_tot = torch.stack(processed_images)

    return {
        'pixel_values': pixel_values_tot,
        'label': torch.tensor(examples['label'])
    }

def train():
    pl.seed_everything(SEED)

    args = SimpleNamespace(
        BATCH_SIZE=32,
        MAX_EPOCHS=20,
        GRADIENT_CLIP_VAL=1.0,
        LR=1e-5,
        WEIGHT_DECAY=0.05,
        NUM_EXPERTS=6,
        TOP_K=2,
        AUX_LOSS_WEIGHT=0.1,
        ROUTER_TEMPERATURE=0.5,
        TRAIN_LAST_N_LAYERS=3,
        UNFREEZE_LAST_N_BLOCKS=0,
        POOLING_STRATEGY="mil",
        NOISE_STREAM=True,
        OUT_CLASSES=3,
        LAMBDA_ENERGY=0.9,
    )
    
    MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
    model = MoEJEPADeepfakeDetector3ClassesMOD(MODEL_NAME, num_experts=args.NUM_EXPERTS, lr=args.LR, 
                                            aux_loss_weight=args.AUX_LOSS_WEIGHT, router_temperature=args.ROUTER_TEMPERATURE, 
                                            top_k = args.TOP_K, weight_decay=args.WEIGHT_DECAY, train_last_n_layers=args.TRAIN_LAST_N_LAYERS, 
                                            unfreeze_last_n_blocks=args.UNFREEZE_LAST_N_BLOCKS, do_noise_stream=args.NOISE_STREAM, 
                                            pooling_strategy=args.POOLING_STRATEGY, out_classes=args.OUT_CLASSES, lambda_energy=args.LAMBDA_ENERGY)
    
    print(model)
    model.train()

    try:
        processor = AutoVideoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except ValueError:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("download dataset...")
    dataset = load_dataset("saberzl/SID_Set")
    print(dataset)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    train_dataset.set_transform(partial(prepare_sidset, processor=processor))
    val_dataset.set_transform(partial(prepare_sidset, processor=processor))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=4, 
                                  prefetch_factor=2, persistent_workers=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4, 
                                prefetch_factor=2, persistent_workers=True, pin_memory=True)

    name_run="<<YOUR RUN NAME>>"

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",             
        dirpath=f"./checkpoints/{name_run}",        
        filename="best-{epoch:02d}",
        save_top_k=1,                 
        mode="max",               
        save_last=False,               
        every_n_epochs=1              
    )

    # Optional Wandb logging
    # wandb_logger = WandbLogger(project="MoE-JEPA", name=name_run)

    # try:
    #     wandb_logger.log_hyperparams(vars(args))
    # except Exception:
    #     try:
    #         wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
    #     except Exception:
    #         pass


    early_stop_callback = EarlyStopping(monitor="val/acc", patience=3, mode="max")

    custom_ddp_strategy = DDPStrategy(
        timeout=timedelta(seconds=7200),
        find_unused_parameters=True,      
        gradient_as_bucket_view=True     
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        strategy=custom_ddp_strategy,
        num_sanity_val_steps=4,
        precision="bf16-mixed",      
        gradient_clip_val=args.GRADIENT_CLIP_VAL,     
        max_epochs=args.MAX_EPOCHS,             
        # logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return 


if __name__ == "__main__":
    train()
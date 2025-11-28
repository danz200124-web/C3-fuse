#!/usr/bin/env python3
"""
C3-Fuse Training Script
Main training loop for the fusion network
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

sys.path.append('..')
from models.c3fuse import build_c3fuse
from models.losses import build_loss
from tools.metrics import MetricTracker, compute_all_metrics
from tools.visualization import plot_training_curves


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def build_optimizer(model: nn.Module, config: dict):
    """Build optimizer"""
    train_config = config['train']
    optimizer_type = train_config['optimizer'].lower()
    lr = train_config['base_lr']
    wd = train_config['weight_decay']

    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=wd, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def build_scheduler(optimizer, config: dict, num_epochs: int):
    """Build learning rate scheduler"""
    train_config = config['train']
    warmup_epochs = train_config.get('warmup_epochs', 5)

    if train_config['lr_scheduler']['type'] == 'cosine':
        min_lr = train_config['lr_scheduler']['min_lr']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=min_lr
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )

    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> dict:
    """Train for one epoch"""
    model.train()
    metric_tracker = MetricTracker()

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].to(device)
        points = batch['points'].to(device)
        uv = batch['uv'].to(device)
        K = batch['K'].to(device)
        T = batch['T'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images, points, uv, K, T, valid_mask, return_intermediates=True)

                # Prepare targets
                targets = {
                    'labels': labels,
                    'points': points,
                    'uv': uv,
                    'valid_mask': valid_mask,
                    'image_shape': (images.shape[2], images.shape[3])
                }

                loss, loss_dict = criterion(outputs, targets)

            # Backward with scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, points, uv, K, T, valid_mask, return_intermediates=True)

            targets = {
                'labels': labels,
                'points': points,
                'uv': uv,
                'valid_mask': valid_mask,
                'image_shape': (images.shape[2], images.shape[3])
            }

            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Update metrics
        metric_tracker.update(loss_dict, n=images.shape[0])

        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return metric_tracker.get_average()


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model"""
    model.eval()
    metric_tracker = MetricTracker()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

        for batch in pbar:
            images = batch['image'].to(device)
            points = batch['points'].to(device)
            uv = batch['uv'].to(device)
            K = batch['K'].to(device)
            T = batch['T'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images, points, uv, K, T, valid_mask, return_intermediates=False)

            targets = {
                'labels': labels,
                'points': points,
                'uv': uv,
                'valid_mask': valid_mask,
                'image_shape': (images.shape[2], images.shape[3])
            }

            loss, loss_dict = criterion(outputs, targets)

            metric_tracker.update(loss_dict, n=images.shape[0])

            # Collect predictions
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    metrics = compute_all_metrics(all_preds, all_labels, num_classes=2)
    metric_tracker.update(metrics)

    return metric_tracker.get_average()


def main():
    parser = argparse.ArgumentParser(description='Train C3-Fuse model')
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file')
    parser.add_argument('--env', type=str, required=True, help='Path to environment config')
    parser.add_argument('--pretrain2d3d', type=str, default=None, help='Pretrained weights')
    parser.add_argument('--log', type=str, default='runs/c3fuse_exp', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Load configs
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.env, 'r') as f:
        env_config = yaml.safe_load(f)

    # Setup
    setup_logging(args.log)
    logging.info(f"Config: {args.cfg}")
    logging.info(f"Log dir: {args.log}")

    # Set seed
    set_seed(env_config['seed']['torch'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Build model
    logging.info("Building model...")
    model = build_c3fuse(config)

    if args.pretrain2d3d:
        logging.info(f"Loading pretrained weights from {args.pretrain2d3d}")
        # Load pretrained weights (implement loading logic)

    model = model.to(device)

    # Build loss
    criterion = build_loss(config)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    num_epochs = config['train']['epochs']
    scheduler = build_scheduler(optimizer, config, num_epochs)

    # Data loaders (placeholder - implement dataset)
    # train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], ...)
    # val_loader = DataLoader(val_dataset, batch_size=config['val']['batch_size'], ...)

    logging.info("Data loaders created")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': [],
        'val_f1': [],
        'lr': []
    }

    # Training loop
    best_miou = 0.0
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        logging.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\n{'='*50}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"{'='*50}")

        # Train
        # train_metrics = train_epoch(
        #     model, train_loader, criterion, optimizer, device, epoch+1,
        #     use_amp=config['train']['use_amp']
        # )
        # logging.info(f"Train - {MetricTracker().get_summary()}")

        # Validate
        # if (epoch + 1) % config['val']['interval'] == 0:
        #     val_metrics = validate(model, val_loader, criterion, device, epoch+1)
        #     logging.info(f"Val - mIoU: {val_metrics['miou']:.4f}, F1: {val_metrics['f1']:.4f}")

        #     # Save best model
        #     if val_metrics['miou'] > best_miou:
        #         best_miou = val_metrics['miou']
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'best_miou': best_miou,
        #         }, os.path.join(args.log, 'best_model.pth'))
        #         logging.info(f"Saved best model with mIoU: {best_miou:.4f}")

        # Step scheduler
        scheduler.step()
        logging.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    logging.info("Training completed!")


if __name__ == '__main__':
    main()

"""
GAMMA Challenge - 训练脚本
完整的训练流程，包含验证、保存checkpoint、早停等
"""
import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent))

from config import cfg
from dataset import get_dataloader
from models.gamma_net import build_model
from utils import (
    set_seed, get_loss_fn, mixup_data, mixup_criterion,
    MetricCalculator, EarlyStopping, AverageMeter
)


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epoch: int,
    device: str,
    use_mixup: bool = True,
) -> dict:
    """训练一个epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    metric_calc = MetricCalculator()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    accumulation_steps = cfg.ACCUMULATION_STEPS
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        fundus = batch["fundus"].to(device)
        oct = batch["oct"].to(device)
        labels = batch["label"].to(device)
        
        # MixUp增强
        if use_mixup and torch.rand(1).item() < cfg.MIXUP_PROB:
            fundus, oct, labels_a, labels_b, lam = mixup_data(
                fundus, oct, labels, alpha=cfg.MIXUP_ALPHA
            )
            
            logits = model(fundus, oct)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            logits = model(fundus, oct)
            loss = criterion(logits, labels)
        
        # 梯度累积
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        # 记录
        loss_meter.update(loss.item() * accumulation_steps, fundus.size(0))
        metric_calc.update(logits.detach(), batch["label"].to(device))
        
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    metrics = metric_calc.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: str,
) -> dict:
    """验证"""
    model.eval()
    
    loss_meter = AverageMeter()
    metric_calc = MetricCalculator()
    
    pbar = tqdm(val_loader, desc="[Validation]")
    
    for batch in pbar:
        fundus = batch["fundus"].to(device)
        oct = batch["oct"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(fundus, oct)
        loss = criterion(logits, labels)
        
        loss_meter.update(loss.item(), fundus.size(0))
        metric_calc.update(logits, labels)
    
    metrics = metric_calc.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    filepath: Path,
):
    """保存checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)
    print(f"[INFO] Checkpoint saved: {filepath}")


def main(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(cfg.SEED)
    
    # 创建目录
    cfg.setup_dirs()
    
    # 设备
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 数据加载
    print("[INFO] Loading data...")
    train_loader = get_dataloader(
        data_dir=cfg.TRAIN_DIR,
        label_file=cfg.TRAIN_LABEL_FILE,
        mode="train",
    )
    val_loader = get_dataloader(
        data_dir=cfg.VAL_DIR,
        label_file=cfg.VAL_LABEL_FILE,
        mode="val",
    )
    
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader.dataset)}")
    
    # 模型
    print("[INFO] Building model...")
    model = build_model(pretrained=cfg.FUNDUS_PRETRAINED)
    model = model.to(device)
    
    # 损失函数
    criterion = get_loss_fn()
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * cfg.EPOCHS // cfg.ACCUMULATION_STEPS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
        anneal_strategy='cos',
    )
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(cfg.LOG_DIR / timestamp)
    
    # 早停
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        mode='max'
    )
    
    best_kappa = 0.0
    
    # 训练循环
    print("[INFO] Starting training...")
    for epoch in range(cfg.EPOCHS):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            epoch, device, use_mixup=cfg.AUGMENTATION
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{cfg.EPOCHS}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Kappa: {train_metrics['kappa']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Kappa: {val_metrics['kappa']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val Per-class Acc: {val_metrics['per_class_acc']}")
        
        # TensorBoard记录
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalars('Kappa', {
            'train': train_metrics['kappa'],
            'val': val_metrics['kappa']
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        
        # 保存最佳模型
        if val_metrics['kappa'] > best_kappa:
            best_kappa = val_metrics['kappa']
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                cfg.CHECKPOINT_DIR / "best_model.pth"
            )
            print(f"  [BEST] New best Kappa: {best_kappa:.4f}")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                cfg.CHECKPOINT_DIR / f"epoch_{epoch+1}.pth"
            )
        
        # 早停检查
        if early_stopping(val_metrics['kappa']):
            print(f"  [INFO] Improved! Counter reset.")
        else:
            print(f"  [INFO] No improvement. Counter: {early_stopping.counter}/{early_stopping.patience}")
        
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    print(f"\n[INFO] Training complete! Best Kappa: {best_kappa:.4f}")
    print(f"[INFO] Best model saved at: {cfg.CHECKPOINT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAMMA Challenge Training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()
    
    # 覆盖配置
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr:
        cfg.LEARNING_RATE = args.lr
    
    main(args)

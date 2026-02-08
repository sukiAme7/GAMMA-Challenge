"""
GAMMA Challenge - 工具函数
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, roc_auc_score
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from config import cfg


def set_seed(seed: int = None):
    """设置随机种子"""
    seed = seed or cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL = -alpha * (1 - p)^gamma * log(p)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: list = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) 未归一化的预测
            targets: (B,) 类别标签
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=torch.tensor(self.alpha, device=logits.device) if self.alpha else None,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        
        return loss.mean()


def get_loss_fn():
    """获取损失函数"""
    if cfg.LOSS_TYPE == "focal":
        return FocalLoss(
            gamma=cfg.FOCAL_GAMMA,
            alpha=cfg.CLASS_WEIGHTS,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    else:
        return nn.CrossEntropyLoss(
            weight=torch.tensor(cfg.CLASS_WEIGHTS) if cfg.CLASS_WEIGHTS else None,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )


def mixup_data(x_fundus, x_oct, y, alpha=0.2):
    """
    MixUp数据增强
    返回混合后的数据和标签权重
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_fundus.size(0)
    index = torch.randperm(batch_size, device=x_fundus.device)
    
    mixed_fundus = lam * x_fundus + (1 - lam) * x_fundus[index]
    mixed_oct = lam * x_oct + (1 - lam) * x_oct[index]
    
    y_a, y_b = y, y[index]
    return mixed_fundus, mixed_oct, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MetricCalculator:
    """评估指标计算器"""
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.preds = []
        self.labels = []
        self.probs = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """更新预测结果"""
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        self.preds.extend(preds.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.probs.extend(probs.detach().cpu().numpy())
    
    def compute(self) -> dict:
        """计算所有指标"""
        preds = np.array(self.preds)
        labels = np.array(self.labels)
        probs = np.array(self.probs)
        
        # Quadratic Weighted Kappa (主要指标)
        kappa = cohen_kappa_score(labels, preds, weights='quadratic')
        
        # Accuracy
        acc = accuracy_score(labels, preds)
        
        # Macro F1
        f1 = f1_score(labels, preds, average='macro')
        
        # Per-class accuracy
        per_class_acc = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc.append((preds[mask] == c).mean())
            else:
                per_class_acc.append(0.0)
        
        # AUC (one-vs-rest)
        try:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
        
        return {
            'kappa': kappa,
            'accuracy': acc,
            'f1_macro': f1,
            'auc': auc,
            'per_class_acc': per_class_acc,
        }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

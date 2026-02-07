"""
GAMMA Challenge - 推理脚本
用于在测试集上进行推理并生成提交文件
"""
import os
import sys
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent))

from config import cfg
from dataset import get_dataloader
from models.gamma_net import build_model
from utils import MetricCalculator


@torch.no_grad()
def inference(
    model,
    test_loader,
    device: str,
    tta: bool = False,
) -> tuple:
    """
    在测试集上进行推理
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        tta: 是否使用TTA (Test Time Augmentation)
    
    Returns:
        sample_ids: 样本ID列表
        predictions: 预测类别
        probabilities: 预测概率
    """
    model.eval()
    
    all_ids = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(test_loader, desc="[Inference]")
    
    for batch in pbar:
        fundus = batch["fundus"].to(device)
        oct = batch["oct"].to(device)
        sample_ids = batch["sample_id"]
        
        if tta:
            # TTA: 原图 + 水平翻转 + 垂直翻转
            logits_list = []
            
            # 原图
            logits_list.append(model(fundus, oct))
            
            # 水平翻转
            fundus_hflip = torch.flip(fundus, dims=[3])
            oct_hflip = torch.flip(oct, dims=[4])
            logits_list.append(model(fundus_hflip, oct_hflip))
            
            # 垂直翻转
            fundus_vflip = torch.flip(fundus, dims=[2])
            oct_vflip = torch.flip(oct, dims=[3])
            logits_list.append(model(fundus_vflip, oct_vflip))
            
            # 平均
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            logits = model(fundus, oct)
        
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_ids.extend(sample_ids)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return all_ids, np.array(all_preds), np.array(all_probs)


def evaluate_on_test(
    model,
    test_loader,
    device: str,
    tta: bool = False,
) -> dict:
    """
    在测试集上评估（如果有标签）
    """
    model.eval()
    metric_calc = MetricCalculator()
    
    pbar = tqdm(test_loader, desc="[Evaluation]")
    
    for batch in pbar:
        fundus = batch["fundus"].to(device)
        oct = batch["oct"].to(device)
        labels = batch["label"].to(device)
        
        if tta:
            logits_list = []
            logits_list.append(model(fundus, oct))
            
            fundus_hflip = torch.flip(fundus, dims=[3])
            oct_hflip = torch.flip(oct, dims=[4])
            logits_list.append(model(fundus_hflip, oct_hflip))
            
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            logits = model(fundus, oct)
        
        metric_calc.update(logits, labels)
    
    return metric_calc.compute()


def main(args):
    """主推理函数"""
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 加载模型
    print(f"[INFO] Loading model from: {args.checkpoint}")
    model = build_model(pretrained=False)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'metrics' in checkpoint:
        print(f"[INFO] Checkpoint metrics: Kappa={checkpoint['metrics'].get('kappa', 'N/A'):.4f}")
    
    # 加载测试数据
    print("[INFO] Loading test data...")
    test_loader = get_dataloader(
        data_dir=cfg.TEST_DIR,
        label_file=cfg.TEST_LABEL_FILE,
        mode="test",
        batch_size=args.batch_size,
    )
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")
    
    # 评估（如果测试集有标签）
    if args.evaluate:
        print("\n[INFO] Evaluating on test set...")
        metrics = evaluate_on_test(model, test_loader, device, tta=args.tta)
        print(f"\nTest Results:")
        print(f"  Kappa: {metrics['kappa']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Per-class Acc: {metrics['per_class_acc']}")
    
    # 生成预测
    if args.output:
        print(f"\n[INFO] Generating predictions...")
        sample_ids, predictions, probabilities = inference(
            model, test_loader, device, tta=args.tta
        )
        
        # 创建提交文件
        results = pd.DataFrame({
            "sample_id": sample_ids,
            "prediction": predictions,
            "prob_0": probabilities[:, 0],
            "prob_1": probabilities[:, 1],
            "prob_2": probabilities[:, 2],
        })
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"[INFO] Predictions saved to: {output_path}")
        
        # 统计预测分布
        print(f"\nPrediction distribution:")
        for c in range(cfg.NUM_CLASSES):
            count = (predictions == c).sum()
            print(f"  Class {c}: {count} ({count/len(predictions)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAMMA Challenge Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(cfg.CHECKPOINT_DIR / "best_model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(cfg.OUTPUT_DIR / "predictions.csv"),
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test Time Augmentation"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on test set (requires labels)"
    )
    
    args = parser.parse_args()
    main(args)

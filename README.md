# GAMMA Challenge - Glaucoma Grading
# 多模态青光眼分级项目

## 项目结构
```
Gamma_Challenge/
├── src/
│   ├── config.py          # 配置参数
│   ├── dataset.py         # 数据加载
│   ├── models/
│   │   ├── __init__.py
│   │   ├── fundus_branch.py   # 2D Fundus分支
│   │   ├── oct_branch.py      # 3D OCT分支
│   │   ├── fusion.py          # 融合模块
│   │   └── gamma_net.py       # 完整模型
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── utils.py           # 工具函数
├── requirements.txt
└── README.md
```

## 环境配置
```bash
pip install -r requirements.txt
```

## 训练
```bash
python src/train.py
```

## 推理
```bash
python src/inference.py --checkpoint best_model.pth --evaluate 
```
## 实验结果 (Test Results)

| 指标 (Metric) | 数值 (Value) |
| :--- | :--- |
| **Kappa** | 0.8117 |
| **Accuracy** | 0.7500 |
| **Precision** | 0.6735 |
| **Recall** | 0.6769 |
| **F1 Macro** | 0.6617 |
| **AUC** | 0.8928 |
### 各类别准确率 (Per-class Accuracy)

| 类别 (Class) | 准确率 (Accuracy) | 说明 |
| :--- | :--- | :--- |
| **Class 0** | 96.08% | Normal (正常) |
| **Class 1** | 32.00% | Early (早期) |
| **Class 2** | 75.00% | Mid/Advanced (中晚期) |
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
python src/inference.py --checkpoint best_model.pth
```

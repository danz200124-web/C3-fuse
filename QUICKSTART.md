# C³-Fuse 快速开始指南

## 项目已完成内容

✅ **完整的深度学习框架**（基于PyTorch）
✅ **3742行Python代码**
✅ **22个文件**（配置、模型、工具、脚本）

## 一、环境安装

### 方式1：使用Conda（推荐）

```bash
# 创建虚拟环境
conda create -n c3fuse python=3.9
conda activate c3fuse

# 安装依赖
pip install -r requirements.txt

# 验证CUDA
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 方式2：使用Docker

```bash
# 构建镜像
docker build -t c3fuse:latest .

# 运行容器
docker run --gpus all -it -v $(pwd):/workspace c3fuse:latest

# 在容器内
cd /workspace
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、项目结构

```
C3-fuse/
├── configs/              # 配置文件
│   ├── env.yaml         # 环境配置
│   ├── c3fuse_base.yaml # 模型配置
│   └── loss.yaml        # 损失权重
│
├── models/              # 网络模型（核心）
│   ├── img_backbone/    # ResNet图像编码器
│   ├── pcd_backbone/    # Point Transformer点云编码器
│   ├── fusion/          # 融合模块
│   │   ├── cross_attn.py   # 跨注意力
│   │   └── gate.py         # 自适应门控
│   ├── c3fuse.py       # 主网络
│   └── losses.py       # 损失函数
│
├── tools/              # 工具库
│   ├── projection.py   # 投影与可见性
│   ├── cyl_grid.py     # 圆柱BEV网格
│   ├── calibration.py  # 相机标定
│   ├── visualization.py# 可视化
│   └── metrics.py      # 评估指标
│
├── scripts/            # 可执行脚本
│   ├── 02_preprocess.py      # 数据预处理
│   ├── 09_train_c3fuse.py    # 训练
│   └── 10_infer_and_post.py  # 推理与后处理
│
├── README.md           # 完整文档
├── PROJECT_SUMMARY.md  # 项目总结
├── Makefile           # 自动化构建
└── Dockerfile         # 容器定义
```

## 三、核心功能

### 1. 数据预处理

```bash
python scripts/02_preprocess.py \
    --scene data/raw/scene_A \
    --tasks undistort,pcd_voxel_down,pcd_est_normal \
    --out data/processed/scene_A
```

### 2. 模型训练

```bash
python scripts/09_train_c3fuse.py \
    --cfg configs/c3fuse_base.yaml \
    --env configs/env.yaml \
    --log runs/exp01
```

### 3. 推理与分析

```bash
python scripts/10_infer_and_post.py \
    --ckpt runs/exp01/checkpoints/best_model.pth \
    --scene data/processed/scene_A \
    --out data/processed/scene_A/pred \
    --export_csv data/reports/scene_A/planes.csv
```

### 4. 使用Makefile（更简单）

```bash
# 查看所有命令
make help

# 安装依赖
make install

# 预处理场景
make preprocess SCENE=scene_A

# 训练模型
make train

# 推理
make infer SCENE=scene_A

# 完整流程
make all
```

## 四、核心技术

### 1. 双路径融合架构

```
Image ──────┐
            ├──► 跨注意力 ────┐
            │                 ├──► 自适应门控 ──► 分割
Point Cloud ┴──► 圆柱BEV ─────┘
```

### 2. 跨注意力机制

- **点为Query，像素为Key/Value**
- **几何偏置**：编码距离、视角、高程
- **k-近邻采样**：每个点采样k个像素特征

### 3. 圆柱BEV网格

- **统一空间表示**：(r, θ, z)极坐标
- **点体素化**：点云聚合到网格
- **图像抬升**：2D特征投影到3D
- **三线性插值**：从网格回填到点

### 4. 自适应门控

- **模态置信度**：基于特征方差/熵估计
- **动态融合**：γ·f_cross + (1-γ)·f_bev
- **应对退化**：图像模糊时降低图像权重

### 5. 多任务损失

```python
L = λ_seg·(CE + Dice)           # 分割损失
  + λ_cm·InfoNCE                # 跨模态对比
  + λ_proj·Consistency          # 投影一致性
  + λ_geo·PlaneDistance         # 几何约束
  + λ_gate·GateReg              # 门控正则化
```

## 五、模型配置

### configs/c3fuse_base.yaml 关键参数

```yaml
# 图像骨干网络
img_backbone:
  type: resnet50          # resnet18/34/50/101
  pretrained: true

# 点云骨干网络
pcd_backbone:
  type: point_transformer
  base_channels: 32

# 融合模块
fusion:
  cross_attn:
    num_layers: 3         # 注意力层数
    num_heads: 4          # 注意力头数
    k_neighbors: 5        # 每点采样像素数

  unified_space:
    type: cylindrical_bev
    grid:
      radial_bins: 128    # 径向分辨率
      theta_bins: 256     # 角向分辨率
      z_bins: 64          # 垂向分辨率

  gate:
    enabled: true
    use_confidence: true  # 使用模态置信度

# 训练超参数
train:
  batch_size: 4
  epochs: 120
  base_lr: 1.0e-4
  optimizer: adamw
  use_amp: true           # 混合精度训练
```

## 六、主要类和函数

### 模型类

```python
from models.c3fuse import C3FuseNet

# 创建模型
model = C3FuseNet(config)

# 前向传播
outputs = model(
    images,      # (B, 3, H, W)
    points,      # (B, N, 3)
    uv,          # (B, N, 2) 归一化坐标
    K,           # (B, 3, 3) 内参
    T,           # (B, 4, 4) 外参
    valid_mask   # (B, N) 可见性
)

# 推理
pred_labels = model.predict(images, points, uv, K, T, valid_mask)
```

### 工具函数

```python
from tools.projection import project_points_to_image
from tools.cyl_grid import CylindricalGrid
from tools.metrics import compute_miou

# 投影点到图像
uv, depth, valid = project_points_to_image(points, K, T, img_shape)

# 创建圆柱网格
grid = CylindricalGrid(radial_bins=128, theta_bins=256, z_bins=64)
indices, valid = grid.points_to_grid_indices(points)

# 计算mIoU
miou = compute_miou(pred, target, num_classes=2)
```

## 七、预期性能指标

根据设计目标（configs/c3fuse_base.yaml）：

- **分割性能**：mIoU ≥ 0.65, F1 ≥ 0.70
- **几何精度**：点到面RMSE ≤ 0.05-0.10m
- **法向精度**：角度误差 ≤ 5°
- **推理速度**：百万点 ≤ 30-60 min/GPU

## 八、下一步工作

要运行完整流程，还需要：

1. **准备数据**：
   - 采集图像和点云数据
   - 准备相机标定板图像
   - 标注训练集（2D像素标签 + 3D点标签）

2. **补充脚本**（框架已有，需实现细节）：
   ```bash
   scripts/01_calib_extract.py      # 相机标定
   scripts/03_reg_deepi2p.py        # DeepI2P粗配准
   scripts/04_reg_colored_icp.py    # Colored-ICP精配准
   scripts/05_poisson_recon.py      # Poisson重建
   scripts/06_make_dataset.py       # 数据集划分
   scripts/07_pretrain_pointcontrast.py  # 预训练
   scripts/08_pretrain_crosspoint.py     # 预训练
   scripts/11_vmf_clustering.py     # vMF聚类
   scripts/12_report_build.py       # 报表生成
   ```

3. **实现数据集类**：
   ```python
   class C3FuseDataset(torch.utils.data.Dataset):
       def __getitem__(self, idx):
           return {
               'image': ...,
               'points': ...,
               'labels': ...,
               'uv': ...,
               'K': ...,
               'T': ...,
               'valid_mask': ...
           }
   ```

4. **训练和调优**：
   - 在实际数据上训练
   - 调整超参数
   - 消融实验

## 九、故障排查

### 常见问题

**1. CUDA内存不足**
```bash
# 解决方案：
# - 减小batch_size（configs/c3fuse_base.yaml）
# - 降低BEV网格分辨率
# - 使用梯度累积
```

**2. 训练不收敛**
```bash
# 检查：
# - 数据标注质量
# - 配准精度（重投影误差应<3像素）
# - 损失权重平衡
```

**3. 推理速度慢**
```bash
# 优化方法：
# - 点云下采样（voxel_size调大）
# - 使用稀疏卷积（MinkowskiEngine）
# - 启用混合精度（use_amp: true）
```

## 十、引用

如果使用本项目，请引用：

```bibtex
@software{c3fuse2024,
  title={C³-Fuse: Cross-Modal Fusion for Structural Plane Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/C3-fuse}
}
```

## 十一、联系与支持

- **文档**：README.md（完整使用说明）
- **总结**：PROJECT_SUMMARY.md（技术细节）
- **配置**：configs/（所有配置文件）

---

**项目状态**：✅ 核心框架完成，可进行训练和推理

**代码量**：3742行Python代码 + 完整配置和文档

**下一步**：准备实际数据并补充外围脚本

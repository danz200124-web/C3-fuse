# C³-Fuse: Cross-Modal Fusion for Robust Structural Plane Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

C³-Fuse is a deep learning framework for robust fusion of images and point clouds for structural geology applications, particularly for detecting and characterizing structural planes (joints, discontinuities) in tunnel faces, rock slopes, and outcrops.

## Features

- **Dual Fusion Paths**: Cross-attention + Cylindrical BEV for complementary strengths
- **Adaptive Gating**: Dynamic modal trust allocation under challenging conditions
- **Geometric Consistency**: Physics-informed losses for planar structures
- **End-to-End Pipeline**: From raw data to engineering reports

## Architecture

```
Image ──────┐
            ├──► Cross-Attention ──┐
            │                      ├──► Adaptive Gate ──► Segmentation ──► Plane Parameters
Point Cloud ┴──► Cylindrical BEV ──┘
```

### Key Components

1. **Cross-Attention Fusion**: Points attend to image features with geometric bias
2. **Cylindrical BEV**: Unified spatial representation for large-scale context
3. **Adaptive Gating**: Balance fusion paths based on modal confidence
4. **Multi-Task Losses**: Segmentation + Contrastive + Projection + Geometric consistency

## Installation

### Requirements

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- 24GB+ GPU recommended

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/C3-fuse.git
cd C3-fuse

# Create conda environment
conda create -n c3fuse python=3.9
conda activate c3fuse

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Docker Installation

```bash
docker build -t c3fuse:latest .
docker run --gpus all -it -v $(pwd):/workspace c3fuse:latest
```

## Project Structure

```
C3-fuse/
├── configs/              # Configuration files
│   ├── env.yaml         # Environment settings
│   ├── c3fuse_base.yaml # Model configuration
│   └── loss.yaml        # Loss weights
├── data/                # Data directory
│   ├── raw/            # Raw acquisitions
│   ├── processed/      # Preprocessed data
│   ├── annotations/    # Labels
│   ├── splits/         # Train/val/test splits
│   └── reports/        # Output reports
├── models/             # Network models
│   ├── img_backbone/   # Image encoders (ResNet)
│   ├── pcd_backbone/   # Point cloud encoders
│   ├── fusion/         # Fusion modules
│   ├── c3fuse.py      # Main model
│   └── losses.py      # Loss functions
├── scripts/            # Executable scripts
│   ├── 01_calib_extract.py      # Camera calibration
│   ├── 02_preprocess.py         # Data preprocessing
│   ├── 09_train_c3fuse.py       # Training
│   └── 10_infer_and_post.py     # Inference + post-processing
├── tools/              # Utilities
│   ├── projection.py   # Projection & visibility
│   ├── cyl_grid.py     # Cylindrical BEV
│   ├── calibration.py  # Camera calibration
│   ├── visualization.py# Visualization tools
│   └── metrics.py      # Evaluation metrics
├── Makefile           # Build automation
├── Dockerfile         # Container definition
└── README.md          # This file
```

## Usage

### 1. Data Preparation

```bash
# Calibrate camera
python scripts/01_calib_extract.py --board 9x6 --square 0.025 \
    --images "data/raw/calib/*.jpg" --out data/raw/meta/cam_intrinsics.yaml

# Preprocess scene
python scripts/02_preprocess.py --scene data/raw/scene_A \
    --tasks undistort,pcd_voxel_down,pcd_est_normal --out data/processed/scene_A
```

### 2. Registration (Cross-Modal Alignment)

```bash
# Coarse registration with DeepI2P
python scripts/03_reg_deepi2p.py --images data/processed/scene_A/images \
    --pcd data/processed/scene_A/pointclouds/scene_A.pcd \
    --intr data/raw/scene_A/meta/cam_intrinsics.yaml \
    --out data/processed/scene_A/meta/extrinsics_init.json

# Fine registration with Colored-ICP
python scripts/04_reg_colored_icp.py \
    --src data/processed/scene_A/pointclouds/scene_A.pcd \
    --init data/processed/scene_A/meta/extrinsics_init.json \
    --out data/processed/scene_A/meta/extrinsics_refined.json
```

### 3. Training

```bash
# Train C3-Fuse model
python scripts/09_train_c3fuse.py \
    --cfg configs/c3fuse_base.yaml \
    --env configs/env.yaml \
    --pretrain2d3d weights/crosspoint.pt \
    --log runs/c3fuse_exp01
```

### 4. Inference & Post-Processing

```bash
# Run inference and extract plane parameters
python scripts/10_infer_and_post.py \
    --ckpt runs/c3fuse_exp01/checkpoints/best_model.pth \
    --scene data/processed/scene_A \
    --out data/processed/scene_A/pred \
    --post ransac_plane_fit \
    --export_csv data/reports/scene_A/planes.csv
```

### 5. Clustering & Reporting

```bash
# Cluster normals using vMF mixture
python scripts/11_vmf_clustering.py \
    --normals data/processed/scene_A/pred/plane_normals.npy \
    --K 3,4,5 --select_by bic --plot \
    --out_fig data/reports/scene_A/stereonet.png

# Generate engineering report
python scripts/12_report_build.py \
    --scene data/processed/scene_A \
    --csv data/reports/scene_A/planes.csv \
    --stn data/reports/scene_A/stereonet.png \
    --out data/reports/scene_A/report.pdf
```

## Configuration

### Model Configuration (`configs/c3fuse_base.yaml`)

Key parameters:
- `img_backbone.type`: Image encoder (resnet18/34/50/101)
- `fusion.cross_attn.num_layers`: Number of cross-attention layers (default: 3)
- `fusion.unified_space.grid`: Cylindrical BEV grid resolution
- `loss.w_*`: Loss weights

### Training Hyperparameters

- Batch size: 4-8 (depends on GPU memory)
- Learning rate: 1e-4 with cosine annealing
- Epochs: 100-120
- Mixed precision: Enabled by default

## Evaluation Metrics

- **Segmentation**: mIoU, F1, Precision, Recall
- **Geometric**: Point-to-plane RMSE, Normal angle error
- **Projection**: 2D-3D consistency

## Advanced Usage

### Ablation Studies

```bash
# Cross-attention only
# Set fusion.unified_space.enabled: false in config

# BEV only
# Set fusion.cross_attn.enabled: false in config

# No gating
# Set fusion.gate.enabled: false in config
```

### Custom Datasets

Implement a custom dataset class inheriting from `torch.utils.data.Dataset`:

```python
class CustomSceneDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'image': ...,      # (3, H, W) tensor
            'points': ...,     # (N, 3) tensor
            'labels': ...,     # (N,) tensor
            'uv': ...,         # (N, 2) normalized coordinates
            'K': ...,          # (3, 3) intrinsic matrix
            'T': ...,          # (4, 4) extrinsic matrix
            'valid_mask': ..., # (N,) boolean mask
        }
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Lower BEV grid resolution

2. **Poor Segmentation Performance**
   - Check registration quality (reprojection error < 3 pixels)
   - Verify data augmentation is enabled
   - Increase training epochs

3. **Slow Inference**
   - Use sparse convolutions (MinkowskiEngine)
   - Reduce point cloud density via voxel downsampling
   - Enable tiling for large scenes

## Citation

If you use C³-Fuse in your research, please cite:

```bibtex
@article{c3fuse2024,
  title={C³-Fuse: Robust Image-Point Cloud Fusion for Structural Plane Detection},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ResNet backbone: [torchvision](https://pytorch.org/vision/)
- Point Transformer: [Point Transformer](https://github.com/POSTECH-CVLab/point-transformer)
- Open3D: [Open3D](http://www.open3d.org/)

## Contact

For questions and feedback:
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/C3-fuse/issues)

---

**Note**: This is a research prototype. For production deployment, additional validation and safety measures are recommended.

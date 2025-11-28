# Makefile for C3-Fuse Project

.PHONY: help install clean test train infer preprocess all

# Variables
PYTHON := python
SCENE := scene_A
CONFIG := configs/c3fuse_base.yaml
ENV_CONFIG := configs/env.yaml

# Help
help:
	@echo "C³-Fuse Makefile Commands:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make preprocess    - Preprocess data for SCENE"
	@echo "  make train         - Train C3-Fuse model"
	@echo "  make infer         - Run inference on SCENE"
	@echo "  make test          - Run unit tests"
	@echo "  make clean         - Clean temporary files"
	@echo "  make all           - Run complete pipeline"
	@echo ""
	@echo "Variables:"
	@echo "  SCENE=scene_A      - Scene name to process"
	@echo "  CONFIG=...         - Model config file"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Verifying CUDA..."
	$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Data preprocessing
preprocess:
	@echo "Preprocessing scene: $(SCENE)"
	mkdir -p data/processed/$(SCENE)
	$(PYTHON) scripts/02_preprocess.py \
		--scene data/raw/$(SCENE) \
		--tasks undistort,pcd_voxel_down,pcd_est_normal \
		--out data/processed/$(SCENE)

# Camera calibration
calibrate:
	@echo "Calibrating camera..."
	mkdir -p data/raw/meta
	$(PYTHON) scripts/01_calib_extract.py \
		--board 9,6 \
		--square 0.025 \
		--images "data/raw/calib/*.jpg" \
		--out data/raw/meta/cam_intrinsics.yaml

# Registration
register:
	@echo "Running registration for $(SCENE)..."
	$(PYTHON) scripts/03_reg_deepi2p.py \
		--images data/processed/$(SCENE)/images \
		--pcd data/processed/$(SCENE)/pointclouds/$(SCENE).pcd \
		--intr data/raw/$(SCENE)/meta/cam_intrinsics.yaml \
		--out data/processed/$(SCENE)/meta/extrinsics_init.json \
		--weights weights/deepi2p.pth
	$(PYTHON) scripts/04_reg_colored_icp.py \
		--src data/processed/$(SCENE)/pointclouds/$(SCENE).pcd \
		--init data/processed/$(SCENE)/meta/extrinsics_init.json \
		--out data/processed/$(SCENE)/meta/extrinsics_refined.json

# Surface reconstruction
reconstruct:
	@echo "Reconstructing surface for $(SCENE)..."
	$(PYTHON) scripts/05_poisson_recon.py \
		--pcd data/processed/$(SCENE)/pointclouds/$(SCENE).pcd \
		--out_mesh data/processed/$(SCENE)/mesh/$(SCENE)_poisson.ply

# Dataset creation
dataset:
	@echo "Creating dataset splits..."
	$(PYTHON) scripts/06_make_dataset.py \
		--scenes data/processed \
		--split_ratio 7,2,1 \
		--out data/splits

# Pre-training
pretrain:
	@echo "Pre-training point cloud backbone..."
	$(PYTHON) scripts/07_pretrain_pointcontrast.py \
		--pcd_root data/processed \
		--epochs 200 \
		--batch_size 8 \
		--lr 1e-3 \
		--save weights/pointcontrast.pt
	@echo "Pre-training cross-modal features..."
	$(PYTHON) scripts/08_pretrain_crosspoint.py \
		--img_root data/processed \
		--pcd_root data/processed \
		--pairs data/splits/train_pairs.txt \
		--epochs 100 \
		--batch_size 8 \
		--lr 5e-4 \
		--init3d weights/pointcontrast.pt \
		--save weights/crosspoint.pt

# Training
train:
	@echo "Training C3-Fuse model..."
	mkdir -p runs
	$(PYTHON) scripts/09_train_c3fuse.py \
		--cfg $(CONFIG) \
		--env $(ENV_CONFIG) \
		--pretrain2d3d weights/crosspoint.pt \
		--log runs/c3fuse_exp

# Inference
infer:
	@echo "Running inference on $(SCENE)..."
	mkdir -p data/processed/$(SCENE)/pred
	mkdir -p data/reports/$(SCENE)
	$(PYTHON) scripts/10_infer_and_post.py \
		--ckpt runs/c3fuse_exp/checkpoints/best_model.pth \
		--scene data/processed/$(SCENE) \
		--out data/processed/$(SCENE)/pred \
		--post ransac_plane_fit \
		--export_csv data/reports/$(SCENE)/planes.csv

# Clustering and reporting
cluster:
	@echo "Clustering plane normals..."
	$(PYTHON) scripts/11_vmf_clustering.py \
		--normals data/processed/$(SCENE)/pred/plane_normals.npy \
		--K 3,4,5 \
		--select_by bic \
		--plot \
		--out_fig data/reports/$(SCENE)/stereonet.png \
		--out data/reports/$(SCENE)/vmf.json

report:
	@echo "Generating engineering report..."
	$(PYTHON) scripts/12_report_build.py \
		--scene data/processed/$(SCENE) \
		--csv data/reports/$(SCENE)/planes.csv \
		--stn data/reports/$(SCENE)/stereonet.png \
		--vmf data/reports/$(SCENE)/vmf.json \
		--out data/reports/$(SCENE)/report.pdf

# Testing
test:
	@echo "Running unit tests..."
	pytest tests/ -v

# Clean
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/

# Deep clean (including outputs)
clean-all: clean
	@echo "Cleaning all outputs..."
	rm -rf runs/
	rm -rf data/processed/
	rm -rf data/reports/

# Complete pipeline
all: preprocess register reconstruct train infer cluster report
	@echo "Complete pipeline finished!"

# Quick demo (single scene)
demo: SCENE=demo
demo:
	@echo "Running demo pipeline..."
	make preprocess SCENE=demo
	make register SCENE=demo
	make infer SCENE=demo
	make cluster SCENE=demo
	@echo "Demo complete! Check data/reports/demo/"

# Docker commands
docker-build:
	docker build -t c3fuse:latest .

docker-run:
	docker run --gpus all -it -v $(PWD):/workspace c3fuse:latest

docker-train:
	docker run --gpus all -v $(PWD):/workspace c3fuse:latest make train

# Environment check
check:
	@echo "Checking environment..."
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
	@$(PYTHON) -c "import open3d; print(f'Open3D: {open3d.__version__}')"
	@echo "Environment OK!"

# C3-Fuse Docker Image
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Install Open3D (if not in requirements)
RUN pip3 install open3d

# Optional: Install MinkowskiEngine for sparse convolutions
# This requires compilation and may take time
# RUN pip3 install ninja
# RUN pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#     --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
#     --install-option="--blas=openblas"

# Copy project files
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /workspace/data/{raw,processed,annotations,splits,reports} \
    /workspace/weights \
    /workspace/runs \
    /workspace/logs

# Verify CUDA installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Default command
CMD ["/bin/bash"]

# Metadata
LABEL maintainer="your.email@example.com"
LABEL description="C3-Fuse: Cross-Modal Fusion for Structural Plane Detection"
LABEL version="1.0"

# Expose ports for tensorboard and jupyter
EXPOSE 6006 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available()" || exit 1

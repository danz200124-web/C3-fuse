@echo off
REM C3-Fuse Environment Setup Script for Windows

echo ========================================
echo C3-Fuse Environment Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda is not installed or not in PATH
    exit /b 1
)

REM Create conda environment
echo Creating conda environment 'c3fuse' with Python 3.9...
conda create -n c3fuse python=3.9 -y
if %errorlevel% neq 0 (
    echo Error: Failed to create conda environment
    exit /b 1
)

REM Install dependencies
echo.
echo Installing project dependencies...
conda run -n c3fuse pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

REM Verify installation
echo.
echo Verifying installation...
conda run -n c3fuse python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the environment, run:
echo     conda activate c3fuse
echo.

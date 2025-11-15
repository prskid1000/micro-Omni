"""
Check setup requirements and provide quick setup guide
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required (current: {}.{})".format(version.major, version.minor))
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not installed (pip install {package_name})")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available (will use CPU - much slower)")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_disk_space(path="data"):
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        print(f"✓ Disk space: {free_gb:.1f} GB free")
        
        if free_gb < 60:
            print(f"  ⚠ Warning: Less than 60GB free. Datasets need ~50-60GB")
            return False
        return True
    except Exception as e:
        print(f"⚠ Could not check disk space: {e}")
        return True

def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files:")
    
    files_to_check = [
        ("data/text/dialogstudio.txt", "DialogStudio text data"),
        ("data/images/annotations.json", "COCO image manifest"),
        ("data/audio/asr.csv", "LibriSpeech ASR CSV"),
    ]
    
    all_exist = True
    for filepath, desc in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024**2)  # MB
            print(f"✓ {desc}: {filepath} ({size:.1f} MB)")
        else:
            print(f"✗ {desc}: {filepath} (not found)")
            all_exist = False
    
    return all_exist

def main():
    print("="*60)
    print("μOmni Setup Checker")
    print("="*60)
    
    print("\n1. Python Version:")
    py_ok = check_python_version()
    
    print("\n2. Required Packages:")
    packages_ok = True
    packages_ok = check_package("torch", "torch") and packages_ok
    packages_ok = check_package("torchaudio", "torchaudio") and packages_ok
    packages_ok = check_package("torchvision", "torchvision") and packages_ok
    packages_ok = check_package("datasets", "datasets") and packages_ok
    packages_ok = check_package("Pillow", "PIL") and packages_ok
    packages_ok = check_package("tqdm", "tqdm") and packages_ok
    packages_ok = check_package("requests", "requests") and packages_ok
    
    print("\n3. CUDA/GPU:")
    cuda_ok = check_cuda()
    
    print("\n4. Disk Space:")
    disk_ok = check_disk_space()
    
    print("\n5. Data Files:")
    data_ok = check_data_files()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if py_ok and packages_ok and disk_ok:
        print("✓ Basic requirements met")
    else:
        print("✗ Some requirements missing")
        print("\nTo install missing packages:")
        print("  pip install torch torchaudio torchvision datasets Pillow tqdm requests")
    
    if cuda_ok:
        print("✓ GPU available - training will be fast")
    else:
        print("⚠ No GPU - training will be very slow (consider using GPU)")
    
    if not data_ok:
        print("\n✗ Data files not found")
        print("\nTo download and format datasets, run:")
        print("  python scripts/download_and_format_datasets.py")
        print("\nOr download manually and convert (see QUICK_START_A_SETUP.md)")
    else:
        print("✓ Data files found - ready to train!")
    
    print("="*60)

if __name__ == "__main__":
    main()


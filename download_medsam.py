#!/usr/bin/env python3
"""
Download MedSAM Checkpoint
Alternative Python method using requests or gdown
"""

import os
import sys
from pathlib import Path
import hashlib

def check_file_exists(filepath: Path) -> bool:
    """Check if file already exists"""
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ {filepath.name} already exists!")
        print(f"   Size: {size_mb:.1f} MB")
        response = input("Download again? (y/n): ")
        if response.lower() != 'y':
            return True
        filepath.unlink()
    return False


def download_with_requests(url: str, output_path: Path):
    """Download using requests library"""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("❌ requests or tqdm not installed")
        print("   Install: pip install requests tqdm")
        return False
    
    print(f"Downloading from: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_with_gdown(file_id: str, output_path: Path):
    """Download from Google Drive using gdown"""
    try:
        import gdown
    except ImportError:
        print("❌ gdown not installed")
        print("   Install: pip install gdown")
        return False
    
    print(f"Downloading from Google Drive (ID: {file_id})")
    
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, str(output_path), quiet=False)
        return True
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def verify_checkpoint(filepath: Path):
    """Verify checkpoint integrity"""
    if not filepath.exists():
        print("❌ File not found!")
        return False
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    
    # MedSAM checkpoint should be around 375 MB
    if size_mb < 300 or size_mb > 500:
        print(f"⚠️ Warning: File size ({size_mb:.1f} MB) seems unusual")
        print("   Expected: ~375 MB")
        print("   File may be corrupted or incomplete")
        return False
    
    print(f"✅ File size looks good: {size_mb:.1f} MB")
    return True


def main():
    print("="*80)
    print("MEDSAM CHECKPOINT DOWNLOAD")
    print("="*80)
    print()
    
    # Output path
    output_path = Path("medsam_vit_b.pth")
    
    # Check if already exists
    if check_file_exists(output_path):
        print("\nCheckpoint ready to use!")
        return
    
    print("MedSAM Information:")
    print("  Repository: https://github.com/bowang-lab/MedSAM")
    print("  Model: SAM ViT-B pretrained on 1.57M medical images")
    print("  Size: ~375 MB")
    print()
    
    # Try different download methods
    print("Download Methods:")
    print("  1. GitHub Releases (recommended)")
    print("  2. Google Drive (if link available)")
    print("  3. Manual download")
    print()
    
    choice = input("Choose method (1-3): ").strip()
    
    success = False
    
    if choice == '1':
        # GitHub releases
        # Note: Update this URL with actual release URL
        url = "https://github.com/bowang-lab/MedSAM/releases/download/v1.0/medsam_vit_b.pth"
        print(f"\nAttempting download from GitHub...")
        print(f"URL: {url}")
        print()
        success = download_with_requests(url, output_path)
    
    elif choice == '2':
        # Google Drive
        print("\nTo download from Google Drive:")
        print("1. Get the file ID from MedSAM repository")
        print("2. Enter it below")
        print()
        file_id = input("Enter Google Drive file ID: ").strip()
        
        if file_id:
            success = download_with_gdown(file_id, output_path)
        else:
            print("❌ No file ID provided")
    
    elif choice == '3':
        # Manual download
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print()
        print("Step 1: Visit MedSAM Repository")
        print("  URL: https://github.com/bowang-lab/MedSAM")
        print()
        print("Step 2: Find Download Link")
        print("  - Check README for download instructions")
        print("  - Look in 'Releases' section")
        print("  - Or find Google Drive link in documentation")
        print()
        print("Step 3: Download medsam_vit_b.pth")
        print("  - Download the checkpoint file")
        print("  - Size should be ~375 MB")
        print()
        print("Step 4: Move to Project Directory")
        print(f"  - Move file to: {Path.cwd()}")
        print(f"  - Final path: {output_path.absolute()}")
        print()
        print("Step 5: Verify")
        print("  - Run this script again to verify download")
        print()
        print("="*80)
        return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Verify download
    if success:
        print()
        if verify_checkpoint(output_path):
            print()
            print("="*80)
            print("✅ DOWNLOAD SUCCESSFUL!")
            print("="*80)
            print(f"  File: {output_path.absolute()}")
            print()
            print("Next steps:")
            print("  1. Ensure config/models/medsam.yaml has:")
            print("     pretrained_source: medsam_vit_b.pth")
            print("  2. Run training with MedSAM model")
            print()
            print("="*80)
        else:
            print()
            print("⚠️ Download may be incomplete or corrupted")
            print("   Please try again or download manually")
    else:
        print()
        print("="*80)
        print("❌ DOWNLOAD FAILED")
        print("="*80)
        print()
        print("Please try manual download:")
        print("  1. Visit: https://github.com/bowang-lab/MedSAM")
        print("  2. Follow their download instructions")
        print(f"  3. Move medsam_vit_b.pth to: {Path.cwd()}")
        print()
        print("="*80)


if __name__ == "__main__":
    main()
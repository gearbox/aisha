#!/usr/bin/env python3
"""
Standalone WAN 2.2 Deployment Script

This script can be run directly without installing the full package.
It downloads the WAN 2.2 GGUF Q8 models and installs the required ComfyUI-GGUF node.

Usage:
    python3 quick_deploy.py
    python3 quick_deploy.py --comfyui /path/to/ComfyUI

Environment Variables:
    COMFYUI_PATH: Path to ComfyUI installation (default: /workspace/ComfyUI)
    HF_TOKEN: Hugging Face token for private repos (optional)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple


class ModelFile(NamedTuple):
    """Model file definition."""

    name: str
    url: str
    filename: str


# WAN 2.2 GGUF Q8 Model Files
WAN_MODELS: list[ModelFile] = [
    ModelFile(
        name="WAN 2.2 High Noise Q8",
        url="https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/HighNoise/dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf",
        filename="dasiwaWAN22I2V14B_midnightflirtHigh-Q8_0.gguf",
    ),
    ModelFile(
        name="WAN 2.2 Low Noise Q8",
        url="https://huggingface.co/Bedovyy/dasiwaWAN22I2V14B-GGUF/resolve/main/LowNoise/dasiwaWAN22I2V14B_midnightflirtLow-Q8_0.gguf",
        filename="dasiwaWAN22I2V14B_midnightflirtLow-Q8_0.gguf",
    ),
]

COMFYUI_GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_status(text: str, success: bool = True) -> None:
    """Print a status message."""
    symbol = "✓" if success else "✗"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {text}")


def download_with_progress(url: str, target_path: Path, hf_token: str | None = None) -> bool:
    """Download a file with progress indication."""
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    # Check if already exists
    if target_path.exists():
        print_status(f"Already exists: {target_path.name}")
        return True

    # Prepare request
    request = urllib.request.Request(url)
    if hf_token:
        request.add_header("Authorization", f"Bearer {hf_token}")

    try:
        print(f"  Downloading: {target_path.name}")

        with urllib.request.urlopen(request, timeout=3600) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8 * 1024 * 1024  # 8MB chunks

            target_path.parent.mkdir(parents=True, exist_ok=True)

            with Path.open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(
                            f"    Progress: {pct:.1f}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="\r"
                        )

            print()  # New line after progress

        # Rename to final location
        temp_path.rename(target_path)
        print_status(f"Downloaded: {target_path.name}")
        return True

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        print_status(f"Failed: {target_path.name} - {e}", success=False)
        return False


def install_custom_node(comfyui_path: Path) -> bool:
    """Install ComfyUI-GGUF custom node."""
    nodes_path = comfyui_path / "custom_nodes"
    node_path = nodes_path / "ComfyUI-GGUF"

    if node_path.exists():
        print_status("ComfyUI-GGUF already installed")
        return True

    print("  Installing ComfyUI-GGUF custom node...")

    try:
        _result = subprocess.run(
            ["git", "clone", COMFYUI_GGUF_REPO, str(node_path)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Install requirements if present
        requirements_file = node_path / "requirements.txt"
        if requirements_file.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                check=True,
            )

        print_status("ComfyUI-GGUF installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        if node_path.exists():
            shutil.rmtree(node_path)
        print_status(f"Failed to install ComfyUI-GGUF: {e.stderr}", success=False)
        return False


def verify_comfyui(comfyui_path: Path) -> bool:
    """Verify ComfyUI installation."""
    required = [
        comfyui_path / "main.py",
        comfyui_path / "models",
        comfyui_path / "custom_nodes",
    ]

    for path in required:
        if not path.exists():
            print_status(f"Missing: {path}", success=False)
            return False

    print_status(f"ComfyUI verified at: {comfyui_path}")
    return True


def deploy_wan_models(comfyui_path: Path, hf_token: str | None = None) -> bool:
    """Deploy WAN 2.2 models to ComfyUI."""
    print_header("WAN 2.2 GGUF Q8 Deployment")

    # Verify ComfyUI
    print("Checking ComfyUI installation...")
    if not verify_comfyui(comfyui_path):
        print("\nComfyUI not found or incomplete!")
        print(f"Expected path: {comfyui_path}")
        return False

    # Install custom node
    print("\nInstalling custom nodes...")
    if not install_custom_node(comfyui_path):
        return False

    # Download models
    print("\nDownloading models...")
    models_path = comfyui_path / "models" / "diffusion_models"
    models_path.mkdir(parents=True, exist_ok=True)

    all_success = True
    for model in WAN_MODELS:
        target = models_path / model.filename
        if not download_with_progress(model.url, target, hf_token):
            all_success = False

    # Summary
    print_header("Deployment Summary")

    if all_success:
        print_status("All models downloaded successfully")
        print_status("ComfyUI-GGUF custom node installed")
        print(f"\nModels location: {models_path}")
        print("\nWAN 2.2 models are ready for use in ComfyUI!")
        print("\nTo use in ComfyUI:")
        print("  1. Use 'UnetLoaderGGUF' node to load the model")
        print("  2. Select one of the dasiwaWAN22I2V14B files")
        print("  3. Connect to your workflow")
    else:
        print_status("Some downloads failed - please retry", success=False)

    return all_success


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy WAN 2.2 GGUF Q8 models to ComfyUI")
    parser.add_argument(
        "--comfyui",
        type=Path,
        default=Path(os.environ.get("COMFYUI_PATH", "/workspace/ComfyUI")),
        help="Path to ComfyUI installation",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (for private repos)",
    )

    args = parser.parse_args()

    success = deploy_wan_models(args.comfyui, args.hf_token)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

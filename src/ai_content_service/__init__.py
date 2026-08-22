"""AI Content Service - Bundle-based deployment automation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .bundle import BundleError, BundleManager
from .comfyui import ComfyUIError, ComfyUIManager
from .config import (
    BundleConfig,
    DeploymentPlan,
    DeployMode,
    Settings,
    get_settings,
)
from .deployer import Deployer, DeploymentError, DeploymentResult
from .downloader import DownloadError, ModelDownloader
from .snapshot import SnapshotError, SnapshotManager
from .workflows import WorkflowError, WorkflowManager

try:
    __version__ = _dist_version("aisha")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "BundleConfig",
    "BundleError",
    "BundleManager",
    "ComfyUIError",
    "ComfyUIManager",
    "DeployMode",
    "Deployer",
    "DeploymentError",
    "DeploymentPlan",
    "DeploymentResult",
    "DownloadError",
    "ModelDownloader",
    "Settings",
    "SnapshotError",
    "SnapshotManager",
    "WorkflowError",
    "WorkflowManager",
    "__version__",
    "get_settings",
]

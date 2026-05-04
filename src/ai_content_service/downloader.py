"""Model downloader for AI Content Service."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path  # noqa: TCH003
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import aiofiles
import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

if TYPE_CHECKING:
    from .config import ModelConfig, ModelFileConfig, Settings

console = Console()


class DownloadError(Exception):
    """Raised when download fails."""

    pass


class ModelDownloader:
    """Async model downloader with progress tracking and verification.

    Supports downloading from:
    - Hugging Face (with optional token for private/gated models)
    - Civitai (with API token)
    - Direct URLs
    """

    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    HF_DOMAINS = ("huggingface.co", "hf.co")
    CIVITAI_DOMAINS = ("civitai.com",)

    def __init__(self, settings: Settings) -> None:
        self._max_concurrent = settings.max_concurrent_downloads
        self._hf_token = settings.hf_token
        self._civitai_token = settings.civitai_api_token
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    async def download_all(
        self,
        models: list[ModelConfig],
        models_base_path: Path,
    ) -> int:
        """Download all models with concurrent limit.

        Returns:
            Number of files successfully downloaded.
        """
        tasks: list[tuple[ModelConfig, ModelFileConfig, Path]] = []

        for model in models:
            model_dir = models_base_path / model.model_type
            if model.subdirectory:
                model_dir = model_dir / model.subdirectory
            model_dir.mkdir(parents=True, exist_ok=True)

            for file in model.files:
                file_path = model_dir / file.filename
                tasks.append((model, file, file_path))

        downloaded = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        ) as progress:

            async def download_with_progress(
                _model: ModelConfig,
                file: ModelFileConfig,
                path: Path,
            ) -> bool:
                async with self._semaphore:
                    task_id = progress.add_task(
                        f"[cyan]{file.filename}",
                        total=file.size_bytes or 0,
                    )
                    try:
                        await self._download_file(file, path, progress, task_id)
                        progress.update(task_id, description=f"[green]✓ {file.filename}")
                        return True
                    except Exception as e:
                        progress.update(task_id, description=f"[red]✗ {file.filename}")
                        console.print(f"[red]Error downloading {file.filename}: {e}[/red]")
                        return False

            results = await asyncio.gather(*[download_with_progress(m, f, p) for m, f, p in tasks])
            downloaded = sum(results)

        return downloaded

    async def _download_file(
        self,
        file: ModelFileConfig,
        path: Path,
        progress: Progress,
        task_id: TaskID,
    ) -> None:
        """Download a single file with progress tracking."""
        # Skip if file exists and matches checksum
        if path.exists() and file.sha256 and await self._verify_checksum(path, file.sha256):
            progress.update(task_id, completed=path.stat().st_size)
            return

        url = self._prepare_download_url(file.url)
        headers = self._get_auth_headers(file.url)

        async with (
            httpx.AsyncClient(follow_redirects=True) as client,
            client.stream("GET", url, headers=headers, timeout=300.0) as response,
        ):
            response.raise_for_status()

            if total := int(response.headers.get("content-length", 0)):
                progress.update(task_id, total=total)

            hasher = hashlib.sha256() if file.sha256 else None

            async with aiofiles.open(path, "wb") as f:
                async for chunk in response.aiter_bytes(self.CHUNK_SIZE):
                    await f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    progress.update(task_id, advance=len(chunk))

            # Verify checksum
            if file.sha256 and hasher:
                actual_hash = hasher.hexdigest()
                if actual_hash != file.sha256:
                    path.unlink()  # Remove corrupted file
                    raise DownloadError(
                        f"Checksum mismatch for {file.filename}: "
                        f"expected {file.sha256}, got {actual_hash}"
                    )

    def _is_civitai_url(self, url: str) -> bool:
        """Return True if the URL's hostname is a Civitai domain."""
        parsed = urlparse(url.lower())
        return any(
            parsed.netloc == domain or parsed.netloc.endswith(f".{domain}")
            for domain in self.CIVITAI_DOMAINS
        )

    def _prepare_download_url(self, url: str) -> str:
        """Prepare URL with authentication tokens if needed."""
        parsed = urlparse(url)

        if self._is_civitai_url(url) and self._civitai_token:
            query = parse_qs(parsed.query)
            query["token"] = [self._civitai_token]
            new_query = urlencode(query, doseq=True)
            return parsed._replace(query=new_query).geturl()

        return url

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        """Return auth headers for the given URL."""
        headers: dict[str, str] = {}
        parsed = urlparse(url)

        if any(domain in parsed.netloc for domain in self.HF_DOMAINS) and self._hf_token:
            headers["Authorization"] = f"Bearer {self._hf_token}"

        return headers

    @staticmethod
    def _parse_content_disposition(header: str | None) -> str | None:
        """Parse filename from Content-Disposition header.

        Prefers filename*= (RFC 5987 UTF-8 encoded) over filename=.
        """
        if not header:
            return None

        # Try filename*= first (UTF-8 encoded, e.g. UTF-8''name%20here)
        for part in header.split(";"):
            part = part.strip()
            if part.lower().startswith("filename*="):
                value = part[len("filename*=") :]
                if "''" in value:
                    _, _, encoded = value.partition("''")
                    from urllib.parse import unquote

                    return unquote(encoded)

        # Fall back to filename=
        for part in header.split(";"):
            part = part.strip()
            if part.lower().startswith("filename="):
                value = part[len("filename=") :].strip()
                if value.startswith('"') and value.endswith('"'):
                    return value[1:-1]
                return value or None

        return None

    async def _verify_checksum(self, path: Path, expected_sha256: str) -> bool:
        """Verify file checksum."""
        hasher = hashlib.sha256()

        async with aiofiles.open(path, "rb") as f:
            while chunk := await f.read(self.CHUNK_SIZE):
                hasher.update(chunk)

        return hasher.hexdigest() == expected_sha256

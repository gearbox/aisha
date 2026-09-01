"""Conservative free-space admission check for ordered provisioning batches."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class HeadroomVerdict:
    """The reproducible result of a batch disk-headroom check."""

    ok: bool
    free_bytes: int
    required_bytes: int
    detail: str


def check_batch_headroom(
    *, models_path: Path, declared_bytes: int, margin: float
) -> HeadroomVerdict:
    """Check free space against gross declared batch bytes plus *margin*.

    ``declared_bytes`` is gross: Aisha does not subtract files that an earlier
    bundle may already have made resident because it does not know the batch's
    file list.  The result is intentionally conservative and can reject a
    batch that reuse would have made fit.  Apex may send a reuse-adjusted
    number when it can calculate one; Aisha does not second-guess it.
    """
    free_bytes = shutil.disk_usage(models_path).free
    required_bytes = int(declared_bytes * (1 + margin))
    ok = free_bytes >= required_bytes
    detail = (
        f"batch disk headroom at {models_path}: {free_bytes} bytes free; "
        f"{required_bytes} bytes required (declared={declared_bytes}, margin={margin:.0%})"
    )
    return HeadroomVerdict(
        ok=ok,
        free_bytes=free_bytes,
        required_bytes=required_bytes,
        detail=detail,
    )

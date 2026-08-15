"""Shared rules for direct-reference (``name @ url``) requirement lines.

Both the snapshot overlay writer (producer) and the ComfyUI requirements
installer (consumer) must agree on which direct references are unusable, so
the predicate lives here rather than in either module.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def is_missing_local_reference(url: str) -> bool:
    """Whether a direct reference is an unavailable local file URL.

    Conda's ``pip freeze`` emits builder-local ``file://`` URLs. They name
    packages already present in the environment, but the source path cannot
    exist on a deployment node and must never be handed back to pip. A
    portable direct reference (git, https, or an existing local path) is not
    matched by this and must be preserved verbatim.
    """
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return False
    if parsed.netloc and parsed.netloc != "localhost":
        path = Path(f"//{parsed.netloc}{unquote(parsed.path)}")
    else:
        path = Path(unquote(parsed.path))
    return not path.exists()

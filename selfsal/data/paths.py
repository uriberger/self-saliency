# Copyright 2026 NVIDIA. Apache-2.0.
"""Where the corpora live.

One helper rather than a hardcoded fallback in each caller. The archive had an absolute
lustre path written into three scripts as a "if it is not here, try there" fallback,
which works on exactly one filesystem and fails silently everywhere else -- the loader
just finds nothing and the stage reports an empty corpus rather than a missing one.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Override with SELFSAL_DATA to point at a shared copy rather than one per checkout.
#: These corpora are large and are not tracked; see docs/install.md.
DATA_ROOT = Path(os.environ.get("SELFSAL_DATA", REPO_ROOT / "data"))


def grpo_sets_dir() -> Path:
    """The held-out validation sets the probes score against."""
    return Path(os.environ.get("SELFSAL_GRPO_SETS", DATA_ROOT / "grpo_sets"))


def require(path: Path, what: str) -> Path:
    """Fail with the path and what it was for, rather than returning an empty result."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Set SELFSAL_DATA (or SELFSAL_GRPO_SETS) to "
            f"point at it; see docs/install.md.")
    return Path(path)

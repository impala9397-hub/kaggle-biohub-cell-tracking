"""Shared test configuration.

Three kinds of tests live here:
  * synthetic CPU tests: always run (no data, no network);
  * builder tests: build kernels from the public base notebook. They skip unless the notebook is present
    (kaggle/fetch_base946.sh, or BASE946_DIR pointing at a directory with notebook.ipynb);
  * organizer-script tests: check runtime anchors against the organizer's predict script. They skip unless
    ORGANIZER_REPO points at a checkout of https://github.com/royerlab/kaggle-cell-tracking-competition.
No test needs competition data.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE_NB = Path(os.environ.get("BASE946_DIR") or (ROOT / "kaggle" / "base946")) / "notebook.ipynb"
HAVE_BASE = BASE_NB.is_file()
ORGANIZER_REPO = Path(os.environ.get("ORGANIZER_REPO") or (ROOT / "third_party" / "kaggle-cell-tracking-competition"))
PREDICT_SCRIPT = ORGANIZER_REPO / "scripts" / "predict_unet_transformer.py"

requires_base = pytest.mark.skipif(not HAVE_BASE, reason="public base notebook not fetched (kaggle/fetch_base946.sh)")


def skip_without_base() -> None:
    """Call at the top of a helper that builds a kernel."""
    if not HAVE_BASE:
        pytest.skip("public base notebook not fetched (kaggle/fetch_base946.sh)")

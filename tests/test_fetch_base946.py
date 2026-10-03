"""kaggle/fetch_base946.sh: it pins the same hash as the builder and installs nothing but the pinned notebook.

No network: the script is run in file mode (a notebook path as argument). The install test needs the public base
notebook (skips without it).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from conftest import BASE_NB, ROOT, requires_base

sys.path.insert(0, str(ROOT / "kaggle"))
import build_r946  # noqa: E402

SCRIPT = ROOT / "kaggle" / "fetch_base946.sh"


def _run(args: list[str], base_dir: str, cwd=None) -> subprocess.CompletedProcess:
    env = {**os.environ, "BASE946_DIR": base_dir}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, cwd=cwd, capture_output=True, text=True)


def test_script_pins_the_builder_hash():
    m = re.search(r'^WANT="([0-9a-f]{64})"$', SCRIPT.read_text(), re.M)
    assert m and m.group(1) == build_r946.BASE946_CODE_SHA256
    assert os.access(SCRIPT, os.X_OK)


def test_other_notebook_is_refused_and_not_installed(tmp_path):
    other = tmp_path / "other.ipynb"
    other.write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["print('another notebook')\n"],
                                            "metadata": {}, "outputs": [], "execution_count": None}],
                                 "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    r = _run([str(other)], str(tmp_path / "base"))
    assert r.returncode != 0 and "MISMATCH" in r.stdout
    assert not (tmp_path / "base").exists()


def test_non_notebook_is_refused(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<html>not a notebook</html>")
    r = _run([str(page)], str(tmp_path / "base"))
    assert r.returncode != 0 and "not a notebook" in r.stderr
    assert not (tmp_path / "base").exists()


@requires_base
def test_pinned_notebook_is_installed_relative_to_cwd(tmp_path):
    r = _run([str(BASE_NB.resolve())], "rel/base946", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert ": OK" in r.stdout
    assert (tmp_path / "rel" / "base946" / "notebook.ipynb").read_bytes() == BASE_NB.read_bytes()

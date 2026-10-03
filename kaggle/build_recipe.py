#!/usr/bin/env python3
"""build_recipe.py — build one of the submitted kernels by name.

    python kaggle/build_recipe.py --list
    python kaggle/build_recipe.py final-headavg-pmax8 out/final      # writes out/final/{notebook.ipynb,kernel-metadata.json}
    python kaggle/build_recipe.py final-headavg-pmax8 --print-env    # show the build_r946.py environment only

A recipe in kaggle/recipes.json is the build_r946.py environment of a submitted kernel. `extends` copies the parent
recipe first; `overrides` is a list of knob-set names or literal {knob: value} dicts, merged in order (a later value
replaces an earlier one in place, so the order of new env lines in the notebook is stable).
The base notebook must be present (kaggle/fetch_base946.sh). Pushing the result needs the owner's private datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECIPES = HERE / "recipes.json"


def load() -> dict:
    return json.loads(RECIPES.read_text())


def resolve(name: str, data: dict | None = None) -> dict:
    """Return {'slug', 'env', 'overrides', 'chain'} for a recipe, following `extends`."""
    data = data or load()
    recipes, knob_sets = data["recipes"], data["knob_sets"]
    if name not in recipes:
        raise SystemExit(f"unknown recipe {name!r}; known: {', '.join(recipes)}")
    chain, cur = [], name
    while cur:
        if cur in chain:
            raise SystemExit(f"recipe cycle at {cur}")
        chain.append(cur)
        cur = recipes[cur].get("extends")
    env: dict[str, str] = {}
    overrides: dict[str, str] = {}
    for rname in reversed(chain):
        r = recipes[rname]
        env.update(r.get("env", {}))
        for item in r.get("overrides", []):
            overrides.update(knob_sets[item] if isinstance(item, str) else item)
    return {"slug": recipes[name]["slug"], "env": env, "overrides": overrides, "chain": list(reversed(chain))}


def build_env(name: str) -> dict[str, str]:
    r = resolve(name)
    env = {"KERNEL_SLUG": r["slug"], **r["env"]}
    if r["overrides"]:
        env["OVERRIDES"] = json.dumps(r["overrides"])
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recipe", nargs="?")
    p.add_argument("dst", nargs="?")
    p.add_argument("--list", action="store_true", help="list recipes")
    p.add_argument("--print-env", action="store_true", help="print the build_r946.py environment and exit")
    a = p.parse_args()
    data = load()
    if a.list or not a.recipe:
        for k, r in data["recipes"].items():
            print(f"{k:22s} {r['description']}")
        return
    env = build_env(a.recipe)
    if a.print_env or not a.dst:
        for k, v in env.items():
            print(f"{k}={v}")
        return
    run_env = {k: v for k, v in os.environ.items()
               if not (k.startswith(("V1284", "TERTIARY_", "NOTEBOOK_PATCH", "SCRIPT_PATCH", "FAST_IO", "BIOHUB_CSV_FLOAT"))
                       or k in ("OVERRIDES", "KERNEL_SLUG", "NO_ROBUSTNESS"))}
    run_env.update(env)
    r = subprocess.run([sys.executable, str(HERE / "build_r946.py"), str(Path(a.dst).resolve())], cwd=ROOT, env=run_env)
    raise SystemExit(r.returncode)  # build_r946.py has already printed its own error message


if __name__ == "__main__":
    main()

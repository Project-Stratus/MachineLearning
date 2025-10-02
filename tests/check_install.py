#!/usr/bin/env python3
"""
Project setup sanity checker.

Run from the repo root (where pyproject.toml lives):

    python scripts/check_project_setup.py
    # optional extras:
    python scripts/check_project_setup.py --build --pip-check --verbose
"""
from __future__ import annotations
import argparse
import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

OK = "✅"
FAIL = "❌"
WARN = "⚠️ "

REPO_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "check_project_setup.py") else Path.cwd()
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"

def echo(msg: str, *, status: str | None = None):
    if status:
        print(f"{status} {msg}")
    else:
        print(msg)

def run(cmd: list[str], **popen_kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **popen_kwargs)

def load_pyproject():
    if not PYPROJECT.exists():
        echo(f"pyproject.toml not found at {PYPROJECT}", status=FAIL); sys.exit(2)
    # Python 3.11+ has tomllib built-in
    try:
        import tomllib
    except Exception:
        echo("Python 3.11+ is required (missing tomllib)", status=FAIL); sys.exit(2)
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)

def parse_requires_python(spec: str | None) -> tuple[int, int] | None:
    # crude parser for formats like ">=3.11" or ">=3.11,<4"
    if not spec:
        return None
    m = re.search(r">=\s*([0-9]+)\.([0-9]+)", spec)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def ensure_no_src_init():
    init = SRC_DIR / "__init__.py"
    if init.exists():
        echo(f"`{init}` exists. With a src/ layout, `src` should NOT be a package.", status=FAIL)
        return False
    return True

def expected_packages_from_pyproject(pp) -> list[Path]:
    try:
        pkgs = pp["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    except KeyError:
        echo("Could not find [tool.hatch.build.targets.wheel].packages in pyproject.toml", status=FAIL)
        sys.exit(2)
    paths = [REPO_ROOT / p for p in pkgs]
    return paths

def check_package_dirs(package_paths: list[Path]) -> bool:
    ok = True
    for p in package_paths:
        if not p.exists():
            echo(f"Package path missing: {p}", status=FAIL); ok = False; continue
        if not p.is_dir():
            echo(f"Package path is not a directory: {p}", status=FAIL); ok = False; continue
        init = p / "__init__.py"
        if not init.exists():
            echo(f"Missing __init__.py in {p} (required for importable package)", status=FAIL); ok = False
        else:
            echo(f"Found package dir and __init__: {p}", status=OK)
    return ok

def imports_for(package_paths: list[Path]) -> list[str]:
    # import names are the last path component (e.g., src/agents -> 'agents')
    return sorted({p.name for p in package_paths})

def check_imports(import_names: list[str]) -> bool:
    ok = True
    # src must NOT be importable
    spec_src = importlib.util.find_spec("src")
    if spec_src:
        echo(f"`src` is importable ({spec_src.origin}) — it should NOT be.", status=FAIL); ok = False
    else:
        echo("`src` is not importable (good).", status=OK)
    for name in import_names:
        spec = importlib.util.find_spec(name)
        if not spec:
            echo(f"Cannot find import '{name}'. Did editable install succeed?", status=FAIL); ok = False
        else:
            echo(f"Import available: {name}  -> {getattr(spec,'origin',None)}", status=OK)
    return ok

# def pip_show(dist_name: str) -> bool:
#     cp = run([sys.executable, "-m", "pip", "show", dist_name])
#     if cp.returncode != 0 or not cp.stdout.strip():
#         echo(f"`pip show {dist_name}` failed or package not installed.\n{cp.stdout}", status=FAIL)
#         return False
#     echo(f"`pip show {dist_name}` ok:\n{textwrap.indent(cp.stdout.strip(), '   ')}", status=OK)
#     return True

def pip_show(dist_name: str, *, verbose: bool = False) -> bool:
    cp = run([sys.executable, "-m", "pip", "show", dist_name])
    if cp.returncode != 0 or not cp.stdout.strip():
        echo(f"`pip show {dist_name}` failed or package not installed.", status=FAIL)
        return False

    name = version = location = None
    for line in cp.stdout.splitlines():
        if line.startswith("Name: "):
            name = line.split("Name: ", 1)[1].strip()
        elif line.startswith("Version: "):
            version = line.split("Version: ", 1)[1].strip()
        elif line.startswith("Location: "):
            location = line.split("Location: ", 1)[1].strip()

    if verbose:
        echo(f"`pip show` ok: {name or dist_name} {version or ''} @ {location or '(unknown)'}", status=OK)
    else:
        echo("Installed package metadata found.", status=OK)
    return True


def pip_check() -> bool:
    cp = run([sys.executable, "-m", "pip", "check"])
    if cp.returncode == 0:
        echo("pip check passed (no dependency conflicts).", status=OK)
        return True
    else:
        echo("pip check reported issues:\n" + textwrap.indent(cp.stdout, "   "), status=WARN)
        return False

def build_wheel_and_inspect(import_names: list[str]) -> bool:
    # Try to build without installing anything. If 'build' isn't present, suggest it.
    cp = run([sys.executable, "-m", "build"], cwd=str(REPO_ROOT))
    if cp.returncode != 0:
        echo("`python -m build` failed. Install 'build' (pip install build) and retry.\n" +
             textwrap.indent(cp.stdout, "   "), status=WARN)
        return False
    # Find latest wheel
    dist = REPO_ROOT / "dist"
    wheels = sorted(dist.glob("*.whl"))
    if not wheels:
        echo("No wheels found under dist/ after build.", status=FAIL); return False
    whl = wheels[-1]
    echo(f"Built wheel: {whl.name}", status=OK)
    ok = True
    with zipfile.ZipFile(whl) as z:
        top_levels = {name.split("/")[0] for name in z.namelist() if "/" in name}
        # Only require that our import_names are present; other files (metadata) will also exist.
        missing = [n for n in import_names if n not in top_levels]
        if missing:
            echo(f"Wheel missing expected top-level packages: {missing}", status=FAIL); ok = False
        else:
            echo(f"Wheel contains expected packages: {sorted(set(import_names))}", status=OK)
    return ok

def main():
    ap = argparse.ArgumentParser(description="Validate src/ packaging & editable install.")
    ap.add_argument("--build", action="store_true", help="Also build a wheel and inspect its contents.")
    ap.add_argument("--pip-check", action="store_true", help="Run `pip check` for dependency conflicts.")
    ap.add_argument("--verbose", action="store_true", help="More logging.")
    args = ap.parse_args()

    echo(f"Repo root: {REPO_ROOT}")
    pp = load_pyproject()

    # Basic pyproject assertions
    project = pp.get("project", {})
    dist_name = project.get("name", "(unknown)")
    requires_py = project.get("requires-python")
    hb = pp.get("build-system", {})
    backend = hb.get("build-backend", "")
    if backend != "hatchling.build":
        echo(f"Unexpected build-backend: {backend!r} (expected 'hatchling.build')", status=FAIL); sys.exit(2)
    else:
        echo("Build backend is hatchling.build", status=OK)

    # Python version check (best effort)
    minver = parse_requires_python(requires_py)
    if minver:
        if sys.version_info < (minver[0], minver[1]):
            echo(f"Python {sys.version.split()[0]} < required {requires_py}", status=FAIL); sys.exit(2)
        else:
            echo(f"Python {sys.version.split()[0]} satisfies requires-python {requires_py}", status=OK)
    else:
        echo("No/unknown requires-python constraint; skipping.", status=WARN)

    # src/ layout checks
    if not SRC_DIR.exists():
        echo(f"`src/` directory not found at {SRC_DIR}", status=FAIL); sys.exit(2)
    else:
        echo("Found src/ directory", status=OK)

    layout_ok = ensure_no_src_init()

    # package list & directory checks
    pkg_paths = expected_packages_from_pyproject(pp)
    if args.verbose:
        echo("Packages from pyproject:\n" + "\n".join(f"  - {p}" for p in pkg_paths))
    pkgs_ok = check_package_dirs(pkg_paths)

    # Editable install check: try imports
    import_names = imports_for(pkg_paths)
    imports_ok = check_imports(import_names)

    # Metadata check
    show_ok = pip_show(dist_name, verbose=args.verbose)

    # Optional checks
    wheel_ok = True
    if args.build:
        wheel_ok = build_wheel_and_inspect(import_names)

    dep_ok = True
    if args.pip_check:
        dep_ok = pip_check()

    all_ok = all([layout_ok, pkgs_ok, imports_ok, show_ok, wheel_ok, dep_ok])
    if all_ok:
        echo("\nAll checks passed. You're good to go! 🎉", status=OK)
        sys.exit(0)
    else:
        echo("\nSome checks failed. See messages above.", status=FAIL)
        sys.exit(1)

if __name__ == "__main__":
    main()

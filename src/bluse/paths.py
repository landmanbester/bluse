"""
Where the workspace lives.

The code is installed; the data is not. A 21 GB pile of HDF5 cannot live inside
a wheel, and the outputs (700 MB of feature matrices, 670 MB of cluster tables)
should not land in site-packages either. So every module resolves its paths at
runtime against a *workspace* -- an ordinary directory holding:

    <workspace>/
        data/           the .h5 stamp files you downloaded          (input)
        catalogues/     Track A output                              (generated)
        features/       Track B feature matrices                    (generated)
        clusters/       Track B cluster tables and plots            (generated)
        plots/          explore.py PNGs                             (generated)
        masks/          empirically derived RFI masks               (generated)

Only `data/` has to exist; the rest are created on demand.

Resolution order, first hit wins:

    1. an explicit --workspace on the command line
    2. $BLUSE_ROOT
    3. the nearest directory at or above the cwd that contains a `data/`,
       searching no further up than the enclosing project (a directory holding
       `.git` or `pyproject.toml`) and never above $HOME
    4. the cwd

Rule 3 is what makes `cd myrun && bluse-track-a` work from anywhere inside that
tree. Its bounds matter: an unbounded search happily climbs out of a checkout
and adopts an unrelated ~/data, which fails silently and looks like it worked.
"""

from __future__ import annotations

import os
import sys
from glob import glob

ENV_VAR = "BLUSE_ROOT"

# Created on demand. `data` is deliberately absent -- we never invent an input
# directory, because an empty one is indistinguishable from a wrong workspace.
OUTPUT_SUBDIRS = ("catalogues", "features", "clusters", "plots", "masks")

_root: str | None = None


def set_workspace(path: str | None) -> None:
    """Pin the workspace. Called by every CLI before defaults are resolved."""
    global _root
    if path:
        _root = os.path.abspath(os.path.expanduser(path))


# Walking up stops at a project boundary. Without this, running from a checkout
# that has no data/ kept climbing and silently adopted an unrelated ~/data as
# the workspace -- the worst possible failure, because everything then "works"
# against the wrong directory.
_BOUNDARY_MARKERS = (".git", "pyproject.toml")


def _discover() -> str:
    start = os.path.abspath(os.getcwd())
    home = os.path.abspath(os.path.expanduser("~"))
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "data")):
            return d
        # Check the boundary directory itself, then stop.
        if any(os.path.exists(os.path.join(d, m)) for m in _BOUNDARY_MARKERS):
            return start
        parent = os.path.dirname(d)
        if parent == d or d == home:
            return start
        d = parent


def _nearby_candidates(limit: int = 5) -> list[str]:
    """Directories one level below the cwd that do look like workspaces."""
    out = []
    try:
        for name in sorted(os.listdir(os.getcwd())):
            if name.startswith(".") or not os.path.isdir(name):
                continue
            if os.path.isdir(os.path.join(name, "data")):
                out.append(name)
    except OSError:
        pass
    return out[:limit]


def workspace() -> str:
    global _root
    if _root is None:
        env = os.environ.get(ENV_VAR)
        _root = os.path.abspath(os.path.expanduser(env)) if env else _discover()
    return _root


def subdir(name: str, create: bool = False) -> str:
    p = os.path.join(workspace(), name)
    if create:
        os.makedirs(p, exist_ok=True)
    return p


def data_dir() -> str:
    return subdir("data")


def require_data_dir() -> str:
    """data_dir(), but exit with something actionable if it is not there."""
    p = data_dir()
    if not os.path.isdir(p):
        msg = [f"No data directory at {p}",
               "",
               f"The workspace resolved to {workspace()} and it has no data/."]
        near = _nearby_candidates()
        if near:
            msg += ["", "These directories below the cwd look like workspaces:"]
            msg += [f"    --workspace {n}" for n in near]
        msg += ["",
                "Otherwise cd into a directory that has a data/, or point at one:",
                "",
                f"    export {ENV_VAR}=/path/to/workspace",
                "    ...or pass --workspace /path/to/workspace",
                "",
                "See the README section 'Getting the data'."]
        sys.exit("\n".join(msg))
    return p


def banner() -> str:
    return f"workspace: {workspace()}"


def add_workspace_arg(parser) -> None:
    """The --workspace flag, worded identically everywhere."""
    parser.add_argument(
        "--workspace", metavar="DIR", default=None,
        help=f"directory holding data/ and the generated outputs. Default: "
             f"${ENV_VAR}, else the nearest parent of the cwd containing "
             f"a data/ directory")


def catalogues_dir() -> str:
    return subdir("catalogues")


def features_dir() -> str:
    return subdir("features")


def clusters_dir() -> str:
    return subdir("clusters")


def plots_dir() -> str:
    return subdir("plots")


def resolve_files(given) -> list[str]:
    """
    Expand user-given HDF5 paths, or default to every .h5 in <workspace>/data.

    A bare name resolves against the data directory, so `bluse-track-a
    sband_short.h5` and `bluse-track-a data/sband_short.h5` mean the same thing
    from anywhere.
    """
    d = data_dir()
    if not given:
        require_data_dir()
        found = sorted(glob(os.path.join(d, "*.h5")))
        if not found:
            sys.exit(f"No .h5 files in {d}. "
                     f"See the README section 'Getting the data'.")
        return found
    out = []
    for p in given:
        if os.path.isdir(p):
            out.extend(sorted(glob(os.path.join(p, "*.h5"))))
        elif os.path.exists(p):
            out.append(p)
        elif os.path.exists(os.path.join(d, p)):
            out.append(os.path.join(d, p))
        else:
            sys.exit(f"Not found: {p}")
    return out

"""
Two suites with different jobs.

tests/unit/       synthetic, deterministic, runs anywhere, gates commits.
tests/workspace/  golden values measured on real feature matrices. Catches a
                  regression in the science, but cannot gate a commit because
                  the data is not in the repository.
"""


import pytest

from bluse import feature_io as FIO
from bluse import paths


def pytest_collection_modifyitems(config, items):
    """Skip workspace-marked tests when there is no features/ to read."""
    try:
        feat = paths.features_dir()
        have = bool(FIO.discover(feat, include_combined=True))
    except Exception:
        have = False
    if have:
        return
    skip = pytest.mark.skip(reason="no BLUSE workspace with features/ found")
    for item in items:
        if "workspace" in item.keywords:
            item.add_marker(skip)

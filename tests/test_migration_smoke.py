"""Smoke test: jxl migrated to next/ml/jxl, key submodules importable."""

import importlib

import pytest


def test_jxl_submodules_importable() -> None:
    """Core submodules must import after migration (no auto/tao — deprecated)."""
    for mod in ["jxl.cls", "jxl.det", "jxl.iqa", "jxl.track", "jxl.model"]:
        importlib.import_module(mod)


def test_deprecated_modules_removed() -> None:
    """auto/tao (deprecated per spec §6) must not exist in migrated copy."""
    for mod in ["jxl.auto", "jxl.tao"]:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)

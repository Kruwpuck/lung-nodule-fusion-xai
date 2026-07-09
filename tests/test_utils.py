"""Tests for src/utils/ and src/models/registry."""
import os
import tempfile

import pytest


def test_cached_nonempty():
    from src.utils.io import cached
    f = tempfile.mktemp()
    open(f, "w").write("x")
    assert cached(f)
    os.unlink(f)


def test_cached_missing():
    from src.utils.io import cached
    assert not cached("/tmp/__nope_lungfuse__")


def test_cached_empty_file():
    from src.utils.io import cached
    f = tempfile.mktemp()
    open(f, "w").close()
    assert not cached(f)
    os.unlink(f)


def test_csvlogger_header_once():
    from src.utils.logger import CSVLogger
    f = tempfile.mktemp(suffix=".csv")
    lg = CSVLogger(f, ["epoch", "loss"])
    lg.log({"epoch": 1, "loss": 0.5})
    lg.log({"epoch": 2, "loss": 0.3})
    lg.close()
    lines = open(f).readlines()
    assert lines[0].strip() == "epoch,loss"
    assert len(lines) == 3
    os.unlink(f)


def test_csvlogger_no_double_header_on_append():
    from src.utils.logger import CSVLogger
    f = tempfile.mktemp(suffix=".csv")
    lg = CSVLogger(f, ["epoch", "loss"])
    lg.log({"epoch": 1, "loss": 0.5})
    lg.close()
    lg2 = CSVLogger(f, ["epoch", "loss"])
    lg2.log({"epoch": 2, "loss": 0.3})
    lg2.close()
    lines = open(f).readlines()
    headers = [l for l in lines if l.startswith("epoch")]
    assert len(headers) == 1, "header written twice"
    os.unlink(f)


def test_fix_seed_runs():
    from src.utils.seed import fix_seed
    fix_seed(0)


def test_registry_name_map_count():
    from src.models.registry import _NAME_MAP
    assert len(_NAME_MAP) == 8


def test_registry_all_names_resolve():
    from src.models.registry import _NAME_MAP
    for k, v in _NAME_MAP.items():
        assert isinstance(v, str) and v

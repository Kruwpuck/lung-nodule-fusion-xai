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


def test_append_row_creates_header():
    from src.utils.logger import append_row
    f = tempfile.mktemp(suffix=".csv")
    append_row(f, {"run_id": "a", "score": 0.1})
    lines = open(f).readlines()
    assert lines[0].strip() == "run_id,score"
    assert len(lines) == 2
    os.unlink(f)


def test_append_row_appends_without_rewriting_header():
    from src.utils.logger import append_row
    f = tempfile.mktemp(suffix=".csv")
    append_row(f, {"run_id": "a", "score": 0.1})
    append_row(f, {"run_id": "b", "score": 0.2})
    lines = open(f).readlines()
    assert len(lines) == 3
    assert lines[0].strip() == "run_id,score"
    os.unlink(f)


def test_append_row_extends_header_for_new_columns(tmp_path=None):
    """Rev1 task 5c: a row with columns the existing runs.csv never had
    (e.g. modality_dropout_rate) must extend the header, not silently drop
    the new data — old rows get backfilled blank for the new column."""
    from src.utils.logger import append_row
    f = tempfile.mktemp(suffix=".csv")
    append_row(f, {"run_id": "a", "score": 0.1})
    append_row(f, {"run_id": "b", "score": 0.2, "modality_dropout_rate": 0.2})
    rows = open(f).read().splitlines()
    assert rows[0] == "run_id,score,modality_dropout_rate"
    assert rows[1] == "a,0.1,"
    assert rows[2] == "b,0.2,0.2"
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

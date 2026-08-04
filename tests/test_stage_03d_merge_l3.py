"""Cek gate Langkah 3 terhadap tabel sintetis.

Yang diuji cuma yang bisa salah diam-diam: penggabungan yang menggandakan baris,
dan gate yang meloloskan hasil buruk. Nilai AUC di sini karangan, bukan hasil nyata.
"""
import os

import pandas as pd
import pytest

from src import stage_03d_merge_l3 as m

# Arm hasil Langkah 1, yang sudah ada di CSV utama sebelum Langkah 3.
LEGACY_ARMS = ["cnn_only", "radiomics_only", "fusion_intermediate",
                "fusion_early", "fusion_late"]
# Yang ditulis run Langkah 3: lima arm di atas plus dua arm rebalanced.
L3_ARMS = LEGACY_ARMS + ["fusion_intermediate_branch_norm", "fusion_intermediate_gmu"]


def _ablation(backbones, suffix="", rad_auc=0.9318, jitter=0.0, arms=None):
    """Tabel ablasi sintetis, satu baris per backbone x fold x arm."""
    arms = L3_ARMS if arms is None else arms
    rows = []
    for b in backbones:
        for fold in range(5):
            for i, arm in enumerate(arms):
                name = arm + suffix if arm.startswith("fusion_intermediate") else arm
                rows.append({
                    "backbone": b, "arm": name, "fold": fold,
                    "fs_method": "mutual_info_classif",
                    # AUC unik per arm supaya G-4 tidak menyala karena data uji.
                    "auc": rad_auc if arm == "radiomics_only" else 0.80 + 0.01 * i + jitter,
                })
    return pd.DataFrame(rows)


def _delong(backbones, suffix=""):
    rows = []
    for b in backbones:
        for arm in ("fusion_intermediate_branch_norm", "fusion_intermediate_gmu"):
            rows.append({
                "backbone": b, "fusion_arm": arm + suffix, "fusion_auc": 0.91,
                "best_single_arm": "radiomics", "best_single_auc": 0.9318,
                "delong_p": 0.01,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Susun artifacts/ lengkap: CSV utama plus kedua direktori kondisi."""
    for d in [m.MAIN_DIR] + list(m.CONDITIONS.values()):
        os.makedirs(tmp_path / d, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    # CSV utama sebelum Langkah 3: hanya arm Langkah 1, nol baris arm rebalanced.
    _ablation(m.BACKBONES, arms=LEGACY_ARMS).to_csv(
        os.path.join(m.MAIN_DIR, "ablation_summary.csv"), index=False)
    _delong(m.BACKBONES).iloc[0:0].to_csv(
        os.path.join(m.MAIN_DIR, "delong_fusion.csv"), index=False)

    for cond, d in m.CONDITIONS.items():
        suffix = "_moddrop_auxloss" if cond == "reg" else ""
        jitter = 0.001 if cond == "reg" else 0.0
        _ablation(m.BACKBONES, suffix, jitter=jitter).to_csv(
            os.path.join(d, "ablation_summary.csv"), index=False)
        _delong(m.BACKBONES, suffix).to_csv(
            os.path.join(d, "delong_fusion.csv"), index=False)
    return tmp_path


def test_merge_adds_exactly_60_rows_and_all_gates_pass(repo):
    m.merge()
    assert m.check() is True

    abl = pd.read_csv(os.path.join(m.MAIN_DIR, "ablation_summary.csv"))
    assert int(m._new_arm_mask(abl["arm"]).sum()) == m.G1_EXPECTED_ROWS


def test_merge_is_idempotent(repo):
    m.merge()
    first = pd.read_csv(os.path.join(m.MAIN_DIR, "ablation_summary.csv")).shape
    m.merge()
    assert pd.read_csv(os.path.join(m.MAIN_DIR, "ablation_summary.csv")).shape == first


def test_merge_never_touches_radiomics_only(repo):
    before = pd.read_csv(os.path.join(m.MAIN_DIR, "ablation_summary.csv"))
    before = before[before.arm == "radiomics_only"].reset_index(drop=True)
    m.merge()
    after = pd.read_csv(os.path.join(m.MAIN_DIR, "ablation_summary.csv"))
    after = after[after.arm == "radiomics_only"].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after[before.columns])


def test_g2_fails_when_a_new_run_drifts(repo):
    """Kontrol negatif harus menyala kalau radiomics_only run baru bergeser."""
    d = m.CONDITIONS["reg"]
    drifted = _ablation(m.BACKBONES, "_moddrop_auxloss",
                        rad_auc=m.G2_TARGET + 2 * m.G2_TOL, jitter=0.001)
    drifted.to_csv(os.path.join(d, "ablation_summary.csv"), index=False)
    m.merge()
    assert m.check() is False


def test_g4_fails_when_conditions_are_identical(repo):
    """AUC identik antar kondisi berarti regularizer tidak benar-benar terpasang."""
    d = m.CONDITIONS["reg"]
    _ablation(m.BACKBONES, "_moddrop_auxloss", jitter=0.0).to_csv(
        os.path.join(d, "ablation_summary.csv"), index=False)
    m.merge()
    assert m.check() is False


def test_check_fails_before_the_runs_exist(repo):
    """Tanpa hasil run, gate harus gagal, bukan lolos karena nol baris."""
    for d in m.CONDITIONS.values():
        os.remove(os.path.join(d, "ablation_summary.csv"))
        os.remove(os.path.join(d, "delong_fusion.csv"))
    assert m.check() is False

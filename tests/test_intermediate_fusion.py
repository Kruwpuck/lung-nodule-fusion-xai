"""Tests for src/fusion/intermediate_fusion.py's training loop, including the
Rev1 task 5c auxiliary per-branch loss wiring (aux_loss_weight -> train_fusion_epoch).
"""
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skip fusion training tests")
import numpy as np
from torch.utils.data import DataLoader

from src.fusion.intermediate_fusion import RadiomicDataset, train_fusion_epoch
from src.models.fusion_net import FusionNet


class _TinyImageDataset:
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(3, 64, 64), idx % 2


def _make_loader(n=8, n_radiomic=10, batch_size=4):
    img_ds = _TinyImageDataset(n)
    rad = np.random.randn(n, n_radiomic).astype(np.float32)
    ds = RadiomicDataset(img_ds, rad)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TestTrainFusionEpochAuxLoss:
    def test_train_epoch_runs_without_aux(self):
        model = FusionNet(n_radiomic=10, backbone_name="mobilenet_v3_large", pretrained=False)
        loader = _make_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        loss = train_fusion_epoch(model, loader, opt, crit, torch.device("cpu"))
        assert loss >= 0

    def test_train_epoch_runs_with_aux_loss(self):
        model = FusionNet(n_radiomic=10, backbone_name="mobilenet_v3_large",
                          pretrained=False, aux_loss_weight=0.5)
        loader = _make_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        loss = train_fusion_epoch(model, loader, opt, crit, torch.device("cpu"))
        assert loss >= 0

    def test_aux_logits_called_once_per_batch_when_weight_positive(self):
        model = FusionNet(n_radiomic=10, backbone_name="mobilenet_v3_large",
                          pretrained=False, aux_loss_weight=0.5)
        calls = []
        orig = model.aux_logits

        def spy():
            calls.append(1)
            return orig()

        model.aux_logits = spy
        loader = _make_loader(n=8, batch_size=4)  # 2 batches
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        train_fusion_epoch(model, loader, opt, crit, torch.device("cpu"))
        assert len(calls) == 2

    def test_aux_logits_never_called_when_weight_zero(self):
        model = FusionNet(n_radiomic=10, backbone_name="mobilenet_v3_large", pretrained=False)

        def spy():
            raise AssertionError("aux_logits() must not be called when aux_loss_weight == 0")

        model.aux_logits = spy
        loader = _make_loader()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        train_fusion_epoch(model, loader, opt, crit, torch.device("cpu"))  # must not raise

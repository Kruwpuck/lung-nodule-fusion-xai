"""Staged unfreezing with frozen BatchNorm, for run03 two-stage fine-tuning.

Four helpers, no new dependencies. They operate on `BackboneClassifier`
(src/models/backbones.py), which wraps every torchvision body as
`features = nn.Sequential(body, AdaptiveAvgPool2d, Flatten)` and keeps the
final `nn.Linear` in `classifier`.

Two traps this module exists to avoid, both of which fail silently:

1. `model.train()` puts BatchNorm back into training mode at the start of every
   epoch. Calling `apply_bn_eval` once before the epoch loop cancels itself on
   the first epoch and nothing complains -- the run just trains with unfrozen
   BN statistics while the log says otherwise. It must be called inside the
   loop, right after `model.train()`. `--self-check` asserts this.

2. Unfreezing by slicing `named_parameters()[-k:]` can open a weight without
   its bias, or half a norm layer. Unfreezing walks child modules instead, so
   a layer is always opened whole.
"""
from __future__ import annotations

import math

import torch.nn as nn


def backbone_body(model: nn.Module) -> nn.Module:
    """The module whose children are the backbone's layer groups.

    `BackboneClassifier.features` is usually `nn.Sequential(body, pool, flatten)`
    where `body` is the torchvision `.features` Sequential (convnext_tiny,
    densenet121, densenet201 -- the three Track 1 backbones). That wrapper
    reports 3 children, so unfreezing 10% of it would open a third of the
    network. When the first child is not itself a Sequential the wrapper is
    already flat (resnet50, googlenet, inception_v3) or is a bare timm model
    (xception, inception_resnet_v2), and `features` is the body.
    """
    features = model.features
    if isinstance(features, nn.Sequential) and len(features) and isinstance(features[0], nn.Sequential):
        return features[0]
    return features


def _head(model: nn.Module) -> nn.Module | None:
    return getattr(model, "classifier", None)


def freeze_all(model: nn.Module) -> int:
    """Set `requires_grad = False` on every parameter. Returns the tensor count."""
    n = 0
    for p in model.parameters():
        p.requires_grad = False
        n += 1
    return n


def unfreeze_top_modules(model: nn.Module, fraction: float) -> tuple[int, int]:
    """Reopen the last `ceil(fraction * n_children)` child modules of the body.

    The classification head is always reopened -- stage 2 keeps training it.
    `fraction = 0.0` is therefore head-only, and `fraction >= 1.0` opens the
    whole body. Returns (n_children_unfrozen, n_children_total) so the caller
    records what was actually opened rather than what was requested.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction!r}")

    children = list(backbone_body(model).children())
    n_total = len(children)
    n_open = n_total if fraction >= 1.0 else math.ceil(fraction * n_total)

    for child in (children[n_total - n_open:] if n_open else []):
        for p in child.parameters():
            p.requires_grad = True

    head = _head(model)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

    return n_open, n_total


def apply_bn_eval(model: nn.Module) -> int:
    """Put every BatchNorm into eval mode. Returns how many were switched.

    Call this INSIDE the epoch loop, immediately after `model.train()`. It
    freezes the running statistics only; affine weight/bias stay trainable if
    their module was unfrozen, which is the intended behaviour -- with batch
    size 16 the running estimates are the unstable part, not the affine
    parameters.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            n += 1
    return n


def build_param_groups(model: nn.Module, head_lr: float, decay_factor: float = 2.6) -> list[dict]:
    """Optimizer param groups with a discriminative learning rate.

    The head gets `head_lr`; each trainable body group, walking backwards from
    the last child, gets the previous group's rate divided by `decay_factor`.
    Frozen children contribute no group and do not consume a decay step, so the
    first unfrozen body group always sits at exactly `head_lr / decay_factor`
    regardless of how deep in the network it happens to be.

    Groups carry a `name` so the per-epoch log can record which rate went where.
    """
    if decay_factor <= 0:
        raise ValueError(f"decay_factor must be positive, got {decay_factor!r}")

    groups: list[dict] = []
    head = _head(model)
    if head is not None:
        head_params = [p for p in head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "name": "head"})

    lr = head_lr
    children = list(backbone_body(model).children())
    for depth, child in enumerate(reversed(children), start=1):
        params = [p for p in child.parameters() if p.requires_grad]
        if not params:
            continue
        lr = lr / decay_factor
        groups.append({"params": params, "lr": lr, "name": f"body[-{depth}]"})

    if not groups:
        raise ValueError("no trainable parameters -- freeze_all ran without any unfreezing")
    return groups


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts, for the provenance CSV."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _self_check() -> None:
    """Assert the silent failures this module exists to prevent."""
    from src.models.backbones import BackboneClassifier

    # pretrained=False: no download, and the weight values are irrelevant here.
    model = BackboneClassifier("densenet121", n_input_channels=3, n_classes=2,
                                pretrained=False, input_size=96)

    def bns():
        return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    # 1. BatchNorm freezing survives model.train(). The control matters: without
    #    it the assert would still pass on a model that never had BN in train
    #    mode to begin with.
    model.train()
    assert any(m.training for m in bns()), \
        "control failed: model.train() left BatchNorm in eval mode"
    n_bn = apply_bn_eval(model)
    assert n_bn > 0, "no BatchNorm found in densenet121 -- wrong probe model"
    assert not any(m.training for m in bns()), \
        "apply_bn_eval left at least one BatchNorm in training mode"
    print(f"[ok] apply_bn_eval switched {n_bn} BatchNorm modules after model.train()")

    # 2. Partial unfreezing is partial: strictly between nothing and everything.
    freeze_all(model)
    trainable, total = count_trainable(model)
    assert trainable == 0, f"freeze_all left {trainable} trainable parameters"

    n_open, n_total = unfreeze_top_modules(model, 0.10)
    trainable, total = count_trainable(model)
    assert 0 < trainable < total, \
        f"unfreeze_top_modules(0.10) gave {trainable}/{total} trainable parameters"
    assert 0 < n_open < n_total, f"opened {n_open} of {n_total} child modules"
    print(f"[ok] unfreeze 10%: {n_open}/{n_total} child modules, "
          f"{trainable:,}/{total:,} parameters trainable")

    # 3. The first unfrozen body group sits exactly one decay step below the head.
    head_lr = 1e-4
    groups = build_param_groups(model, head_lr, decay_factor=2.6)
    assert groups[0]["name"] == "head" and groups[0]["lr"] == head_lr, \
        f"first group is {groups[0]['name']!r} at lr {groups[0]['lr']}"
    assert len(groups) > 1, "10% unfreeze produced no body group"
    expected = head_lr / 2.6
    assert abs(groups[1]["lr"] - expected) < 1e-12, \
        f"first body group lr {groups[1]['lr']} != head_lr/2.6 = {expected}"
    print("[ok] param groups: " + ", ".join(f"{g['name']}@{g['lr']:.3g}" for g in groups))

    # 4. The two extremes behave: 0.0 is head-only, 1.0 opens everything.
    freeze_all(model)
    unfreeze_top_modules(model, 0.0)
    head_only, _ = count_trainable(model)
    expected_head = sum(p.numel() for p in _head(model).parameters())
    assert head_only == expected_head, \
        f"fraction 0.0 left {head_only} trainable, expected the head's {expected_head}"

    freeze_all(model)
    unfreeze_top_modules(model, 1.0)
    trainable, total = count_trainable(model)
    assert trainable == total, f"fraction 1.0 left {total - trainable} parameters frozen"
    print(f"[ok] extremes: 0.0 -> head only ({head_only:,}), 1.0 -> all ({total:,})")

    print("[SELF-CHECK PASSED]")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true",
                    help="run the asserts that guard the two silent failures")
    args = p.parse_args()
    if args.self_check:
        _self_check()
    else:
        p.print_help()


if __name__ == "__main__":
    main()

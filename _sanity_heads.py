
import torch
from src.models.registry import build_model
cfg = {"data": {"n_slices": 3}}
for task, ncls in [("binary",2),("ordinal",1),("grade4",4)]:
    m = build_model("mobilenetv3_small", cfg, task=task)
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    assert y.shape[-1] == ncls, (task, y.shape)
    print("OK", task, tuple(y.shape))
print("ALL SANITY PASS")

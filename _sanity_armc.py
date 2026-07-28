
import torch, pandas as pd
from src.models.registry import build_model
from src.stage_03_train import _TASK_CFG, _filter_for_task

cfg = {"data": {"n_slices": 3}}
m = build_model("mobilenetv3_small", cfg, task="grade3")
x = torch.randn(2, 3, 64, 64)
y = m(x)
assert y.shape[-1] == 3, y.shape
print("OK grade3 head", tuple(y.shape))

df = pd.read_csv("artifacts/patches/labels.csv")
filt = _filter_for_task(df, "grade3")
print("grade3 filtered rows:", len(filt))
print("grade3 value counts:", filt["grade3"].value_counts().to_dict())
assert (filt["grade3"] != -1).all()

from src.stage_04_evaluate import run
print("stage_04 import OK")
print("ALL SANITY PASS")

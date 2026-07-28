
import torch, numpy as np, yaml
from src.models.registry import build_model
from src.evaluation.metrics import ordinal_metrics, derive_binary, derive_grade3

cfg = yaml.safe_load(open("configs/config.yaml"))

for task in ["binary", "ordinal", "grade4"]:
    m = build_model("mobilenetv3_small", cfg, task=task)
    m.eval()
    x = torch.randn(2, cfg["data"]["n_slices"], 64, 64)
    with torch.no_grad():
        out = m(x)
    print(f"task={task:8s} output shape={tuple(out.shape)}")

# bad task should raise
try:
    build_model("mobilenetv3_small", cfg, task="bogus")
    print("FAIL: bad task did not raise")
except ValueError as e:
    print("OK: bad task raised:", e)

# metrics sanity
y_true = np.array([1,2,3,3,4,5,2,1])
y_pred = np.array([1,2,2,3,4,5,3,2])
print("ordinal_metrics:", ordinal_metrics(y_true, y_pred))

y_true_bin = np.array([1,2,3,3,4,5])
y_pred_bin = np.array([1.2,2.1,3.0,2.9,3.8,4.9])
print("derive_binary:", derive_binary(y_true_bin, y_pred_bin))

t3, p3 = derive_grade3(y_true, y_pred)
print("derive_grade3 t:", t3.tolist(), "p:", p3.tolist())

print("ALL_SANITY_OK")

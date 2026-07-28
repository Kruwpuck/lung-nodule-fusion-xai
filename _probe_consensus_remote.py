
import os, sys, inspect, traceback
sys.path.insert(0, os.getcwd())
import numpy as np, pylidc as pl
from pylidc.utils import consensus
print("pylidc version:", getattr(pl, "__version__", "unknown"))
print("consensus sig:", str(inspect.signature(consensus)))
scans = pl.query(pl.Scan).all()
print("scans:", len(scans))
# find first scan with a median!=3 group, try consensus, print real exception
done = 0
for scan in scans:
    try:
        groups = scan.cluster_annotations()
    except Exception as e:
        continue
    for anns in groups:
        if len(anns) < 1:
            continue
        ratings = [a.malignancy for a in anns]
        med = float(np.median(ratings))
        if med == 3:
            continue
        try:
            out = consensus(anns, clevel=0.5, pad=((0,0),(0,0),(0,0)))
            print("OK consensus returned type:", type(out), "len:", (len(out) if hasattr(out,"__len__") else "n/a"))
            if hasattr(out, "__len__"):
                print("  arity:", len(out), "elem types:", [type(x).__name__ for x in out])
            m = out[0]
            print("  mask sum:", int(np.asarray(m).sum()))
        except Exception as e:
            print("CONSENSUS FAILED:", repr(e))
            traceback.print_exc()
        # also test to_volume on this scan
        try:
            v = scan.to_volume(verbose=False)
            print("to_volume OK shape:", v.shape)
        except Exception as e:
            print("TO_VOLUME FAILED:", repr(e))
        done += 1
        break
    if done >= 3:
        break

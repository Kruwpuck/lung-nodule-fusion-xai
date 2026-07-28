
import os, sys, inspect, numpy as np
sys.path.insert(0, os.getcwd())
import pylidc as pl
from pylidc.utils import consensus
from pylidc import Annotation
# print bbox source lines 395-415
import inspect as ins
src = ins.getsource(Annotation.bbox)
print("=== Annotation.bbox source ===")
print(src)
# try consensus with several pad formats on one group
scans = pl.query(pl.Scan).all()
for scan in scans:
    groups = scan.cluster_annotations()
    for anns in groups:
        if len(anns) < 1: continue
        for pad in [None, 0, 512, [(0,0),(0,0),(0,0)]]:
            try:
                out = consensus(anns, clevel=0.5, pad=pad)
                print("pad=", repr(pad), "OK arity", len(out), "masksum", int(np.asarray(out[0]).sum()))
            except Exception as e:
                print("pad=", repr(pad), "FAIL", repr(e))
        raise SystemExit

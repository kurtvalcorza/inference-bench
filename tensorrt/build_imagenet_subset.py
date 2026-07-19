#!/usr/bin/env python3
"""Build a balanced, representative ImageNet-1k validation subset (5 images per
class, all 1000 classes) from the ungated HF mirror `Tsomaros/Imagenet-1k_validation`
(standard 1000-class labels, no token needed).

Usage: python build_imagenet_subset.py [OUT_DIR]
  OUT_DIR defaults to $BENCH_ROOT/vision/inet_val (or ./inet_val).

Writes <OUT_DIR>/{00000.JPEG,...} and <OUT_DIR>/val_map.txt, which the ResNet-50
runs (`trt_mlperf_run.sh`, reference notebooks) consume via --dataset-path.

Needs: pip install "huggingface_hub>=0.24,<1.0" pyarrow   # hub 1.x breaks transformers 4.48
"""
import os, sys
from collections import Counter
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "Tsomaros/Imagenet-1k_validation"        # class-sorted, 50/class
root = os.environ.get("BENCH_ROOT", ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "vision", "inet_val")
os.makedirs(OUT, exist_ok=True)

files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
g = k = 0
counts = Counter()
with open(os.path.join(OUT, "val_map.txt"), "w") as vm:   # context-managed: no leaked/partial handle
    for fn in files:
        t = pq.read_table(hf_hub_download(REPO, fn, repo_type="dataset"))
        for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            if g % 10 == 0:                          # every 10th row => 5 per class, all 1000 classes
                b = img.get("bytes") if isinstance(img, dict) else None
                if b:                                # skip rows that store a path instead of inline bytes
                    with open(os.path.join(OUT, f"{k:05d}.JPEG"), "wb") as im:
                        im.write(b)
                    vm.write(f"{k:05d}.JPEG {int(lab)}\n")
                    counts[int(lab)] += 1
                    k += 1
            g += 1
print(f"kept {k} images -> {OUT}")

# Sanity gate: the "5/class, all 1000 classes" guarantee only holds if the mirror is exactly 50
# contiguous rows per class. If its layout ever changes, the stride desyncs from class boundaries
# and the subset is silently unbalanced (skewing top-1). Fail loudly instead of returning garbage.
per_class = set(counts.values())
if len(counts) != 1000 or per_class != {5}:
    sys.exit(f"ERROR: expected 5 images x 1000 classes (=5000); got {k} images across "
             f"{len(counts)} classes, per-class counts={sorted(per_class)}. The HF mirror layout "
             f"likely changed — fix the sampling in build_imagenet_subset.py, or point DATA=... at a "
             f"known-good val set with a val_map.txt.")

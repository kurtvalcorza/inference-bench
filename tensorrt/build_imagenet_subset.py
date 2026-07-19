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
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "Tsomaros/Imagenet-1k_validation"        # class-sorted, 50/class
root = os.environ.get("BENCH_ROOT", ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "vision", "inet_val")
os.makedirs(OUT, exist_ok=True)

files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
g = k = 0
vm = open(os.path.join(OUT, "val_map.txt"), "w")
for fn in files:
    t = pq.read_table(hf_hub_download(REPO, fn, repo_type="dataset"))
    for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
        if g % 10 == 0:                          # every 10th row => 5 per class, all 1000 classes
            open(os.path.join(OUT, f"{k:05d}.JPEG"), "wb").write(img["bytes"])
            vm.write(f"{k:05d}.JPEG {int(lab)}\n")
            k += 1
        g += 1
vm.close()
print(f"kept {k} images -> {OUT}")

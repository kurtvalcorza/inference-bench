#!/usr/bin/env python3
"""Build a balanced, representative ImageNet-1k validation subset (5 images per
class, all 1000 classes) from the ungated HF mirror `Tsomaros/Imagenet-1k_validation`
(standard 1000-class labels, no token needed).

Usage: python build_imagenet_subset.py [OUT_DIR]
  OUT_DIR defaults to $BENCH_ROOT/vision/inet_val (or ./inet_val).

Writes <OUT_DIR>/{00000.JPEG,...} and <OUT_DIR>/val_map.txt, which the ResNet-50
runs (`trt_mlperf_run.sh`, reference notebooks) consume via --dataset-path.

Atomic: the subset is built in a temp sibling dir and validated BEFORE being moved
into place, so a failed/partial download never leaves a poisoned val_map.txt that a
later run would silently accept.

Needs: pip install "huggingface_hub>=0.24,<1.0" pyarrow   # hub 1.x breaks transformers 4.48
"""
import os, sys, shutil, tempfile
from collections import Counter

REPO = "Tsomaros/Imagenet-1k_validation"        # class-sorted, 50/class


def validate_counts(counts):
    """Return an error string if the subset isn't exactly 5 images x 1000 classes, else None.

    The '5/class, all 1000 classes' guarantee only holds if the mirror is exactly 50
    contiguous rows per class. If its layout changes, the stride desyncs from class
    boundaries and the subset is silently unbalanced (skewing top-1). Importable/testable.
    """
    per_class = set(counts.values())
    if len(counts) != 1000 or per_class != {5}:
        return (f"expected 5 images x 1000 classes (=5000); got {sum(counts.values())} images across "
                f"{len(counts)} classes, per-class counts={sorted(per_class)}. The HF mirror layout "
                f"likely changed — fix the sampling in build_imagenet_subset.py, or point DATA=... at a "
                f"known-good val set with a val_map.txt.")
    return None


def build(out_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out = os.path.abspath(out_dir)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # unique temp sibling (same filesystem => atomic os.replace) so concurrent builds can't collide
    tmp = tempfile.mkdtemp(prefix=os.path.basename(out) + ".building.", dir=os.path.dirname(out) or ".")

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    g = k = 0
    counts = Counter()
    try:
        with open(os.path.join(tmp, "val_map.txt"), "w") as vm:
            for fn in files:
                t = pq.read_table(hf_hub_download(REPO, fn, repo_type="dataset"))
                for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                    if g % 10 == 0:               # every 10th row => 5 per class, all 1000 classes
                        b = img.get("bytes") if isinstance(img, dict) else None
                        if b:                     # skip rows that store a path instead of inline bytes
                            with open(os.path.join(tmp, f"{k:05d}.JPEG"), "wb") as im:
                                im.write(b)
                            vm.write(f"{k:05d}.JPEG {int(lab)}\n")
                            counts[int(lab)] += 1
                            k += 1
                    g += 1

        err = validate_counts(counts)
        if err:
            sys.exit("ERROR: " + err)

        # Cross-check every val_map entry resolves to a real, non-empty file before publishing.
        with open(os.path.join(tmp, "val_map.txt")) as f:
            for ln in f:
                name = ln.split()[0]
                p = os.path.join(tmp, name)
                if not (os.path.isfile(p) and os.path.getsize(p) > 0):
                    sys.exit(f"ERROR: val_map references missing/empty image {name} — aborting.")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)    # never leave a partial temp dir behind
        raise

    # Atomic-ish publish: replace OUT only now that TMP is complete and validated.
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print(f"kept {k} images -> {out}")


def main():
    root = os.environ.get("BENCH_ROOT", ".")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "vision", "inet_val")
    build(out_dir)


if __name__ == "__main__":
    main()

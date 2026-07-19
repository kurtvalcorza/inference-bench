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
import hashlib, os, sys, shutil, tempfile
from collections import Counter

REPO = "Tsomaros/Imagenet-1k_validation"        # class-sorted, 50/class
# Pin the dataset revision (finding #4: an unpinned mutable HF dataset could change content under a
# stable-looking val_map). Override with DATASET_REVISION to use a different snapshot deliberately.
REVISION = os.environ.get("DATASET_REVISION", "55405c49dece42420e68ddd5f80174f19b29ebaf")


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

    # Pin the revision on BOTH the file listing and every download, so the built subset is tied to
    # one immutable dataset snapshot (finding #4).
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
                   if f.endswith(".parquet"))
    print(f"dataset {REPO}@{REVISION[:12]} — {len(files)} parquet shard(s)")
    g = k = 0
    counts = Counter()
    try:
        with open(os.path.join(tmp, "val_map.txt"), "w") as vm:
            for fn in files:
                t = pq.read_table(hf_hub_download(REPO, fn, repo_type="dataset", revision=REVISION))
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

        # Cross-check every val_map entry resolves to a real, non-empty file, and record a content
        # manifest (sha256 of each image + label) so downstream bundles can attest to image CONTENT,
        # not just the val_map (finding #4). A root hash over the sorted per-file lines gives one
        # deterministic digest for the whole subset.
        lines = []
        with open(os.path.join(tmp, "val_map.txt")) as f:
            rows = [ln.split() for ln in f if ln.strip()]
        for name, lab in rows:
            p = os.path.join(tmp, name)
            if not (os.path.isfile(p) and os.path.getsize(p) > 0):
                sys.exit(f"ERROR: val_map references missing/empty image {name} — aborting.")
            with open(p, "rb") as im:
                h = hashlib.sha256(im.read()).hexdigest()
            lines.append(f"{h}  {name}  {lab}")
        lines.sort()
        root = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
        with open(os.path.join(tmp, "dataset_manifest.txt"), "w", encoding="utf-8") as mf:
            mf.write(f"# {REPO}@{REVISION}\n# root sha256: {root}\n")
            mf.write("\n".join(lines) + "\n")
        print(f"content manifest root sha256: {root}")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)    # never leave a partial temp dir behind
        raise

    # Atomic backup-and-rollback publish (finding #7): keep the existing good dataset until the swap
    # succeeds. os.replace can't overwrite a non-empty dir on POSIX, so exchange via renames: move OUT
    # aside, move TMP into place, then drop the backup. If the second rename fails, restore OUT.
    backup = None
    if os.path.exists(out):
        backup = out + ".old." + os.path.basename(tmp).rsplit(".", 1)[-1]
        os.replace(out, backup)               # atomic move of the known-good set out of the way
    try:
        os.replace(tmp, out)                  # atomic move of the validated set into place
    except BaseException:
        if backup is not None:
            os.replace(backup, out)           # roll back to the known-good set
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    print(f"kept {k} images -> {out}")


def main():
    root = os.environ.get("BENCH_ROOT", ".")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "vision", "inet_val")
    build(out_dir)


if __name__ == "__main__":
    main()

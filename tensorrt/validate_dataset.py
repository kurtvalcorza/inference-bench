#!/usr/bin/env python3
"""Independent dataset-profile validation for the ResNet-50 accuracy pass.

Finding #1 (self-certifying partial datasets): the runner previously accepted any
val_map whose entries named a non-empty file, and then derived the EXPECTED sample
count from that *same* map. A truncated or duplicate-heavy map could therefore pass
both the existence check and the `total == EXPECTED` accuracy cross-check — e.g. a
one-image map yields 100% top-1 and "passes".

This module enforces an INDEPENDENT profile before any of the map's own numbers are
trusted: a sample-count floor, a distinct-class floor, unique image paths, and
in-range integer labels. A partial/poisoned set fails loudly here instead of being
scored. It is importable so the checks are unit-tested (tests/test_dataset_validation.py)
as real failure modes, not asserted as source strings.

Floors are env-configurable so the runner works for both documented datasets:
  * representative HF mirror  -> 5,000 images / 1,000 classes  (MIN_SAMPLES=5000 MIN_CLASSES=1000)
  * Imagenette (10-class)     -> 3,925 images / 10 classes
Defaults (MIN_SAMPLES=1000, MIN_CLASSES=10) accept both while rejecting the
one-image / few-image self-certification attack.

Usage (invoked by trt_mlperf_run.sh):
  MIN_SAMPLES=5000 MIN_CLASSES=1000 python validate_dataset.py <DATA_DIR> <val_map.txt>
"""
from __future__ import annotations

import os
import sys


def parse_val_map(text):
    """Return a list of whitespace-split rows from val_map text, skipping blank lines."""
    return [ln.split() for ln in text.splitlines() if ln.strip()]


def validate_profile(rows, min_samples, min_classes, num_classes=1000):
    """Return an error string if `rows` don't meet the required profile, else None.

    All checks are independent of the scorer's own sample count, so a truncated map
    cannot lower the bar it is measured against:
      - every row is well-formed 'path label'
      - labels parse as integers in [0, num_classes)
      - image paths are unique (a duplicate could pad a single favorable class)
      - at least `min_samples` rows total
      - at least `min_classes` distinct labels (diversity/balance floor)
    """
    if not rows:
        return "val_map is empty"
    paths, labels = [], []
    for i, r in enumerate(rows):
        if len(r) < 2:
            return f"row {i} malformed (expected 'path label'): {r!r}"
        p, lab = r[0], r[1]
        try:
            li = int(lab)
        except ValueError:
            return f"row {i} label is not an integer: {lab!r}"
        if not (0 <= li < num_classes):
            return f"row {i} label {li} out of range [0,{num_classes})"
        paths.append(p)
        labels.append(li)
    if len(set(paths)) != len(paths):
        seen, dups = set(), []
        for p in paths:
            if p in seen and p not in dups:
                dups.append(p)
            seen.add(p)
        return (f"duplicate image paths in val_map (e.g. {dups[:3]}) — a partial set "
                f"padded to look complete cannot be trusted")
    if len(paths) < min_samples:
        return (f"only {len(paths)} samples (< floor {min_samples}) — refusing a truncated "
                f"set that could self-certify accuracy over a favorable subset")
    distinct = len(set(labels))
    if distinct < min_classes:
        return (f"only {distinct} distinct classes (< floor {min_classes}) — set is not "
                f"class-diverse enough to trust top-1")
    return None


def missing_or_empty(rows, data_dir):
    """Return the list of row paths that are missing or empty on disk under data_dir."""
    bad = []
    for r in rows:
        p = os.path.join(data_dir, r[0])
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            bad.append(r[0])
    return bad


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: validate_dataset.py <DATA_DIR> <val_map.txt>")
    data, vmap = argv[1], argv[2]
    min_samples = int(os.environ.get("MIN_SAMPLES", "1000"))
    min_classes = int(os.environ.get("MIN_CLASSES", "10"))
    with open(vmap, encoding="utf-8", errors="replace") as f:
        rows = parse_val_map(f.read())

    err = validate_profile(rows, min_samples, min_classes)
    if err:
        sys.exit("dataset profile check FAILED: " + err)

    bad = missing_or_empty(rows, data)
    if bad:
        sys.exit(f"{len(bad)}/{len(rows)} val_map images missing/empty (e.g. {bad[:3]})")

    classes = len({r[1] for r in rows})
    print(f"dataset OK: {len(rows)} images, {classes} classes, all present "
          f"(floors: >={min_samples} samples, >={min_classes} classes)")


if __name__ == "__main__":
    main(sys.argv)

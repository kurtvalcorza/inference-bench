#!/usr/bin/env python3
"""Independent dataset-profile validation for the ResNet-50 accuracy pass.

Finding #1 (self-certifying partial datasets): the runner previously accepted any
val_map whose entries named a non-empty file, and then derived the EXPECTED sample
count from that *same* map. A truncated, duplicate-heavy, or class-skewed map could
therefore pass both the existence check and the `total == EXPECTED` accuracy
cross-check — e.g. a one-image map yields 100% top-1, or 991 easy images from one
class plus one from nine others still "passes" a distinct-class count.

This module enforces an INDEPENDENT profile before any of the map's own numbers are
trusted: a sample-count floor, a distinct-class floor, a **class-balance cap** (no
single class may dominate), unique image paths, in-range integer labels, and
**path containment** (no absolute paths or `..` escaping the dataset dir — finding
#2). A partial/poisoned/cherry-picked set fails loudly here instead of being scored.
It is importable so the checks are unit-tested (tests/test_validate_dataset.py) as
real failure modes, not asserted as source strings.

Floors/caps are env-configurable so the runner works for its documented datasets:
  * representative HF mirror  -> 5,000 images / 1,000 classes  (the runner's default:
    MIN_SAMPLES=5000 MIN_CLASSES=1000)
  * Imagenette (10-class)     -> 3,925 images / 10 classes  (needs explicit overrides
    MIN_SAMPLES=3000 MIN_CLASSES=10)
The class-balance cap (MAX_CLASS_FRACTION, default 0.5) rejects the 991-of-one-class
cherry-pick regardless of the count/class floors.

Usage (invoked by trt_mlperf_run.sh):
  MIN_SAMPLES=5000 MIN_CLASSES=1000 python validate_dataset.py <DATA_DIR> <val_map.txt>
"""
from __future__ import annotations

import os
import sys
from collections import Counter


def parse_val_map(text):
    """Return a list of whitespace-split rows from val_map text, skipping blank lines."""
    return [ln.split() for ln in text.splitlines() if ln.strip()]


def _unsafe_path(p):
    """True if a val_map path is absolute or uses parent traversal (could escape DATA)."""
    if os.path.isabs(p):
        return True
    # normalize separators so a Windows-style '..\\x' is caught on any OS
    parts = p.replace("\\", "/").split("/")
    return ".." in parts or "" in parts[:1]  # leading '/' already caught by isabs; guard empties


def validate_profile(rows, min_samples, min_classes, num_classes=1000, max_class_fraction=0.5):
    """Return an error string if `rows` don't meet the required profile, else None.

    All checks are independent of the scorer's own sample count, so a truncated or
    skewed map cannot lower the bar it is measured against:
      - every row is well-formed 'path label'
      - image paths are RELATIVE and stay under the dataset dir (no abs / '..' escape)
      - labels parse as integers in [0, num_classes)
      - image paths are unique (a duplicate could pad a single favorable class)
      - at least `min_samples` rows total
      - at least `min_classes` distinct labels
      - no single class exceeds `max_class_fraction` of all samples (balance cap)
    """
    if not rows:
        return "val_map is empty"
    paths, labels = [], []
    for i, r in enumerate(rows):
        if len(r) < 2:
            return f"row {i} malformed (expected 'path label'): {r!r}"
        p, lab = r[0], r[1]
        if _unsafe_path(p):
            return (f"row {i} path {p!r} is absolute or uses '..' — val_map paths must be "
                    f"relative and stay under the dataset dir (no escaping DATA)")
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
    counts = Counter(labels)
    distinct = len(counts)
    if distinct < min_classes:
        return (f"only {distinct} distinct classes (< floor {min_classes}) — set is not "
                f"class-diverse enough to trust top-1")
    top = max(counts.values())
    if top > max_class_fraction * len(paths):
        return (f"class imbalance: the largest class holds {top}/{len(paths)} "
                f"({top / len(paths):.0%}) of samples, over the {max_class_fraction:.0%} cap — "
                f"a cherry-picked/unbalanced set can't be trusted for top-1")
    return None


def missing_or_empty(rows, data_dir):
    """Return row paths that are missing, empty, or resolve OUTSIDE the dataset dir.

    Defense in depth for finding #2: even after the string-level `_unsafe_path` check,
    resolve each path and require it to remain under the real dataset root (catches
    symlink escapes), then confirm the file exists and is non-empty.
    """
    root = os.path.realpath(data_dir)
    bad = []
    for r in rows:
        full = os.path.realpath(os.path.join(data_dir, r[0]))
        try:
            contained = os.path.commonpath([root, full]) == root
        except ValueError:
            contained = False  # different drive/root
        if not (contained and os.path.isfile(full) and os.path.getsize(full) > 0):
            bad.append(r[0])
    return bad


def main(argv):
    if len(argv) < 3:
        sys.exit("usage: validate_dataset.py <DATA_DIR> <val_map.txt>")
    data, vmap = argv[1], argv[2]
    min_samples = int(os.environ.get("MIN_SAMPLES", "1000"))
    min_classes = int(os.environ.get("MIN_CLASSES", "10"))
    max_class_fraction = float(os.environ.get("MAX_CLASS_FRACTION", "0.5"))
    with open(vmap, encoding="utf-8", errors="replace") as f:
        rows = parse_val_map(f.read())

    err = validate_profile(rows, min_samples, min_classes, max_class_fraction=max_class_fraction)
    if err:
        sys.exit("dataset profile check FAILED: " + err)

    bad = missing_or_empty(rows, data)
    if bad:
        sys.exit(f"{len(bad)}/{len(rows)} val_map images missing/empty/out-of-dir (e.g. {bad[:3]})")

    classes = len({r[1] for r in rows})
    print(f"dataset OK: {len(rows)} images, {classes} classes, all present & contained "
          f"(floors: >={min_samples} samples, >={min_classes} classes, <= {max_class_fraction:.0%}/class)")


if __name__ == "__main__":
    main(sys.argv)

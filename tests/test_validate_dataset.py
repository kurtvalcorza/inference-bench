"""Finding #1 behavioral: the independent dataset-profile check must reject the
self-certification attacks (truncated map, duplicate padding, bad labels) that a
mere file-existence check would let score a favorable subset as 100% accurate.

These are REAL failure-mode tests against the importable validator, not source-string
assertions — an ineffective implementation cannot satisfy them.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tensorrt"))
from validate_dataset import parse_val_map, validate_profile  # noqa: E402


def _rows(n, classes):
    """n unique images spread across `classes` labels."""
    return [[f"{i:05d}.JPEG", str(i % classes)] for i in range(n)]


def test_full_balanced_set_passes():
    assert validate_profile(_rows(5000, 1000), min_samples=5000, min_classes=1000) is None


def test_imagenette_shape_passes_default_floors():
    # 3925 images / 10 classes should pass the default (1000/10) floors.
    assert validate_profile(_rows(3925, 10), min_samples=1000, min_classes=10) is None


def test_one_image_map_rejected():
    # The core attack: a single favorable image would score 100% — must be rejected by the count floor.
    err = validate_profile(_rows(1, 1), min_samples=1000, min_classes=10)
    assert err is not None and "floor" in err


def test_truncated_set_rejected():
    err = validate_profile(_rows(50, 10), min_samples=1000, min_classes=10)
    assert err is not None and "samples" in err


def test_duplicate_paths_rejected():
    rows = _rows(2000, 50)
    rows[7][0] = rows[0][0]  # duplicate a path to pad a class
    err = validate_profile(rows, min_samples=1000, min_classes=10)
    assert err is not None and "duplicate" in err.lower()


def test_too_few_classes_rejected():
    # Enough samples but only 3 distinct classes — not diverse enough to trust top-1.
    err = validate_profile(_rows(2000, 3), min_samples=1000, min_classes=10)
    assert err is not None and "classes" in err


def test_noninteger_label_rejected():
    rows = _rows(2000, 50)
    rows[1][1] = "cat"
    assert validate_profile(rows, min_samples=1000, min_classes=10) is not None


def test_out_of_range_label_rejected():
    rows = _rows(2000, 50)
    rows[2][1] = "5000"  # >= num_classes
    assert validate_profile(rows, min_samples=1000, min_classes=10) is not None


def test_malformed_row_rejected():
    rows = _rows(2000, 50)
    rows[3] = ["only_a_path.JPEG"]  # missing label
    assert validate_profile(rows, min_samples=1000, min_classes=10) is not None


def test_parse_skips_blank_lines():
    assert parse_val_map("a.JPEG 0\n\n  \nb.JPEG 1\n") == [["a.JPEG", "0"], ["b.JPEG", "1"]]

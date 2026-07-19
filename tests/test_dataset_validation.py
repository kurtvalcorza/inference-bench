"""Finding #4: the subset balance gate must reject any non-5x1000 layout."""
import sys, pathlib
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tensorrt"))
from build_imagenet_subset import validate_counts  # noqa: E402


def test_balanced_5x1000_passes():
    assert validate_counts(Counter({i: 5 for i in range(1000)})) is None


def test_missing_class_rejected():
    assert validate_counts(Counter({i: 5 for i in range(999)})) is not None


def test_skewed_class_rejected():
    c = Counter({i: 5 for i in range(999)}); c[999] = 7
    assert validate_counts(c) is not None


def test_empty_rejected():
    assert validate_counts(Counter()) is not None

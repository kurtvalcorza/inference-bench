"""Behavioral test (finding #11) for the atomic subset builder (#4/#6): with a fake
HF data source that yields an UNBALANCED set, build() must raise and leave NO poisoned
output dir and NO leftover temp dir — i.e. a failed build can't be silently trusted."""
import sys, types, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tensorrt"))


def _install_fake_hf(monkeypatch, n_rows, label):
    """Fake huggingface_hub + pyarrow.parquet yielding n_rows all of one label (=> unbalanced)."""
    hf = types.ModuleType("huggingface_hub")
    hf.HfApi = type("FakeApi", (), {"list_repo_files": lambda self, r, repo_type: ["a.parquet"]})
    hf.hf_hub_download = lambda repo, fn, repo_type: "ignored"

    class _Col:
        def __init__(self, v): self._v = v
        def to_pylist(self): return self._v

    class _Tbl:
        def column(self, name):
            if name == "image":
                return _Col([{"bytes": b"jpegbytes"} for _ in range(n_rows)])
            return _Col([label] * n_rows)

    pa = types.ModuleType("pyarrow")
    paq = types.ModuleType("pyarrow.parquet")
    paq.read_table = lambda p: _Tbl()
    pa.parquet = paq
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", paq)


def test_unbalanced_build_leaves_no_poison(monkeypatch, tmp_path):
    _install_fake_hf(monkeypatch, n_rows=30, label=0)   # every 10th => 3 imgs, all class 0 => unbalanced
    import build_imagenet_subset as b

    out = tmp_path / "inet_val"
    with pytest.raises(SystemExit):          # validate_counts rejects (not 5x1000)
        b.build(str(out))

    assert not out.exists(), "failed build must NOT publish a (poisoned) output dir"
    leftovers = list(tmp_path.glob("*.building*"))
    assert not leftovers, f"temp build dir must be cleaned up, found {leftovers}"

"""Finding #3: the run-bundle wrapper must capture a well-formed manifest and
propagate the wrapped command's real exit code."""
import json, shutil, subprocess, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts" / "run_bundle.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def _latest_bundle(label):
    d = ROOT / "results" / "bundles"
    cands = sorted(d.glob(f"*-{label}"))
    return cands[-1] if cands else None


def _cleanup():
    shutil.rmtree(ROOT / "results" / "bundles", ignore_errors=True)


def test_manifest_and_pass():
    _cleanup()
    try:
        r = subprocess.run(["bash", str(BUNDLE), "utpass", "--", "bash", "-c", "echo hi; exit 0"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0
        b = _latest_bundle("utpass")
        assert b is not None
        m = json.loads((b / "manifest.json").read_text())
        for key in ("label", "utc", "exit_status", "passed", "command", "meta", "files"):
            assert key in m
        assert m["exit_status"] == 0 and m["passed"] is True
        assert (b / "run.log").exists() and (b / "env.txt").exists()
    finally:
        _cleanup()


def test_exit_code_propagates():
    _cleanup()
    try:
        r = subprocess.run(["bash", str(BUNDLE), "utfail", "--", "bash", "-c", "echo boom; exit 7"],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 7, "wrapper must return the wrapped command's real exit code"
        b = _latest_bundle("utfail")
        assert json.loads((b / "manifest.json").read_text())["exit_status"] == 7
    finally:
        _cleanup()

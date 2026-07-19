"""Finding #3: the run-bundle wrapper must capture a well-formed manifest and
propagate the wrapped command's real exit code.

IMPORTANT: these tests point run_bundle.sh at an isolated RESULTS_ROOT (a fresh
tempdir) so they NEVER touch the real results/bundles/ directory — running the
suite must not delete on-hardware benchmark evidence.
"""
import json, os, shutil, subprocess, tempfile, pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts" / "run_bundle.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")


def _run(label, inner_cmd):
    """Run the wrapper with an isolated RESULTS_ROOT; return (returncode, bundle_dir)."""
    tmp = tempfile.mkdtemp(prefix="rb_test_")
    try:
        env = {**os.environ, "RESULTS_ROOT": tmp}
        r = subprocess.run(["bash", str(BUNDLE), label, "--", "bash", "-c", inner_cmd],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        # RESULTS_ROOT is isolated per test, so any bundle dir created here is ours.
        bundles = sorted(p for p in pathlib.Path(tmp).iterdir() if p.is_dir())
        bundle = bundles[-1] if bundles else None
        # read what we need before cleanup
        manifest = json.loads((bundle / "manifest.json").read_text()) if bundle else None
        has_logs = bool(bundle and (bundle / "run.log").exists() and (bundle / "env.txt").exists())
        return r.returncode, manifest, has_logs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # only ever removes this test's own tempdir


def test_manifest_and_pass():
    rc, m, has_logs = _run("utpass", "echo hi; exit 0")
    assert rc == 0
    assert m is not None
    for key in ("label", "utc", "exit_status", "passed", "command", "meta", "files"):
        assert key in m
    assert m["exit_status"] == 0 and m["passed"] is True
    assert has_logs


def test_exit_code_propagates():
    rc, m, _ = _run("utfail", "echo boom; exit 7")
    assert rc == 7, "wrapper must return the wrapped command's real exit code"
    assert m is not None and m["exit_status"] == 7 and m["passed"] is False


def test_label_is_sanitized():
    """A label with path separators must not escape RESULTS_ROOT (the '/' is the traversal vector)."""
    rc, m, _ = _run("a/b/../evil", "echo x; exit 0")
    assert rc == 0 and m is not None, "bundle must be created inside RESULTS_ROOT, not escape it"
    assert "/" not in m["label"], "path separators must be stripped from the label"

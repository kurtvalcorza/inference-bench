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

# Skip on native Windows: run_bundle.sh is a POSIX script and the only bash on PATH is WSL's, which
# receives Windows-style paths it can't resolve (exit 127). These run in CI on Linux and locally in
# the WSL distro — the two POSIX environments where the script is actually used.
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="POSIX bash with matching paths required (runs in Linux CI / WSL, not native Windows)",
)


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


def test_no_stale_trt_logs_attached():
    """Finding #2 behavioral: a command that creates NO new TRT run dir must not have a pre-existing,
    unrelated run's logs copied into its bundle."""
    fake_bench = tempfile.mkdtemp(prefix="rb_bench_")
    tmp_results = tempfile.mkdtemp(prefix="rb_res_")
    try:
        # Plant a pre-existing TRT run dir with a marker file, BEFORE the wrapped command runs.
        preexisting = pathlib.Path(fake_bench) / "vision" / "runs" / "20200101-000000.aaaaaa"
        preexisting.mkdir(parents=True)
        (preexisting / "mlperf_log_summary.txt").write_text("STALE — must not be attached\n")

        env = {**os.environ, "RESULTS_ROOT": tmp_results, "BENCH_ROOT": fake_bench}
        # Wrap `true`: it creates no new run dir, so nothing under runs/ should be attached.
        subprocess.run(["bash", str(BUNDLE), "stale", "--", "true"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
        bundles = sorted(p for p in pathlib.Path(tmp_results).iterdir() if p.is_dir())
        assert bundles, "a bundle dir should still be created"
        b = bundles[-1]
        attached = b / "tensorrt_run"
        assert not attached.exists(), "no tensorrt_run/ should be attached when the command created no run dir"
        assert (b / "tensorrt_run.note").exists(), "should note that no TRT run dir was created"
    finally:
        shutil.rmtree(fake_bench, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)


def test_marker_run_dir_is_attached_by_identity():
    """Finding #4: a wrapped command that reports its run dir via BUNDLE_RUNROOT_FILE gets exactly
    that dir attached — by identity, not a global before/after diff."""
    fake_bench = tempfile.mkdtemp(prefix="rb_bench_")
    tmp_results = tempfile.mkdtemp(prefix="rb_res_")
    try:
        # This inner command mimics the runner: create a unique run dir and record it in the marker.
        inner = (
            'd="$BENCH_ROOT/vision/runs/mine.$$"; mkdir -p "$d"; '
            'echo hello > "$d/mlperf_log_summary.txt"; '
            'printf "%s\\n" "$d" > "$BUNDLE_RUNROOT_FILE"; echo ran'
        )
        env = {**os.environ, "RESULTS_ROOT": tmp_results, "BENCH_ROOT": fake_bench}
        subprocess.run(["bash", str(BUNDLE), "marker", "--", "bash", "-c", inner],
                       cwd=ROOT, capture_output=True, text=True, env=env)
        b = sorted(p for p in pathlib.Path(tmp_results).iterdir() if p.is_dir())[-1]
        copied = list((b / "tensorrt_run").rglob("mlperf_log_summary.txt"))
        assert copied, "the run dir named in the marker must be copied into tensorrt_run/"
        assert copied[0].read_text().strip() == "hello"
        assert not (b / ".trt_runroot").exists(), "the marker file itself must not linger in the bundle"
    finally:
        shutil.rmtree(fake_bench, ignore_errors=True)
        shutil.rmtree(tmp_results, ignore_errors=True)

"""Finding #9 (behavioral, not source-scan): exercise the TRT runner's REAL
`check_valid()` gate against fake LoadGen summaries. Instead of asserting that the
string "Result is : VALID" appears in the script, we extract the actual function
body from trt_mlperf_run.sh and drive it through its failure modes — so an
ineffective implementation (e.g. one that forgets to set FAILED) cannot pass.
"""
import os
import shutil
import subprocess
import tempfile
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tensorrt" / "trt_mlperf_run.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="POSIX bash required (runs in Linux CI / WSL, not native Windows)",
)

VALID_SUMMARY = """MLPerf Results Summary
Scenario : SingleStream
90.0th percentile latency (ns) : 5000000
Result is : VALID
"""
INVALID_SUMMARY = """MLPerf Results Summary
Scenario : SingleStream
90.0th percentile latency (ns) : 5000000
Result is : INVALID
  Min duration satisfied : NO
"""


def _run_gate(summary, rc):
    """Extract check_valid() from the real runner, drive it once, return FAILED (0/1)."""
    out = tempfile.mkdtemp(prefix="gate_")
    try:
        if summary is not None:
            with open(os.path.join(out, "mlperf_log_summary.txt"), "w") as f:
                f.write(summary)
        # a run.log so the failure branches' `tail` has something to read
        with open(os.path.join(out, "run.log"), "w") as f:
            f.write("(fake run log)\n")
        # Pull the actual check_valid function body out of the script and source it, so we test the
        # shipped gate — not a copy. The function spans `check_valid () {` .. a line that is just `}`.
        script = f'''
set -u
eval "$(awk '/^check_valid \\(\\) \\{{/,/^\\}}/' "{RUNNER.as_posix()}")"
FAILED=0
check_valid TestScenario "{pathlib.Path(out).as_posix()}" {rc} >/dev/null 2>&1
echo "$FAILED"
'''
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        return r.stdout.strip()
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_valid_summary_rc0_passes():
    assert _run_gate(VALID_SUMMARY, 0) == "0", "a VALID summary with rc 0 must NOT set FAILED"


def test_invalid_summary_fails():
    assert _run_gate(INVALID_SUMMARY, 0) == "1", "a non-VALID LoadGen result must set FAILED"


def test_missing_summary_fails():
    assert _run_gate(None, 0) == "1", "a missing summary must set FAILED (can't claim a pass)"


def test_nonzero_rc_fails():
    assert _run_gate(VALID_SUMMARY, 7) == "1", "a non-zero harness exit must set FAILED even if a summary exists"

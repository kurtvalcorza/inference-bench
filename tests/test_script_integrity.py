"""Source-scan regression guards for findings #2, #5, #6, #7 — cheap checks that the
integrity fixes stay in place (no GPU/network needed)."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8", errors="ignore")


# --- #2 fail-loudly ---------------------------------------------------------
def test_trt_runner_verifies_valid_and_propagates_failure():
    s = read("tensorrt/trt_mlperf_run.sh")
    assert "Result is : VALID" in s, "perf runs must be gated on a parsed VALID result"
    assert "FAILED" in s and "exit 1" in s, "runner must be able to exit non-zero on failure"
    # the old blanket-suppress on the perf run must be gone (accuracy run may still use || true)
    assert "run_main SingleStream" in s and "|| true" not in s.split("Accuracy")[0].split("run_main SingleStream")[1]


def test_polygraphy_propagates_failure():
    s = read("standards/polygraphy_resnet.sh")
    assert "polygraphy run FAILED" in s and "exit 1" in s
    parse_failure = s.split("could not parse the average inference time", 1)[1]
    assert "exit 1" in parse_failure, "an unparseable result must not produce a successful bundle"


def test_bundle_records_dataset_balance_knob():
    s = read("scripts/run_bundle.sh")
    env_block = s.split("for v in", 1)[1].split("; do", 1)[0]
    assert "MAX_CLASS_FRACTION" in env_block


# --- #5 TensorRT status checks ---------------------------------------------
def test_backend_checks_trt_status():
    s = read("tensorrt/backend_tensorrt.py")
    assert "returned None" in s, "must None-check build/deserialize/context"
    assert "execute_async_v3 failed" in s, "must raise if execute_async_v3 returns False"


def test_gpu_bench_checks_execute():
    s = read("microbench/gpu_bench.py")
    assert "execute_async_v3 failed" in s


# --- #6 safe load + pinning -------------------------------------------------
def test_export_prefers_safe_load():
    s = read("tensorrt/export_resnet50_onnx.py")
    assert "weights_only=True" in s
    assert "ALLOW_UNSAFE_PICKLE" in s, "unsafe pickle load must be opt-in"


def test_harness_commit_pinned():
    assert "INFERENCE_REF" in read("tensorrt/trt_mlperf_run.sh")


# --- #7 physical-core label -------------------------------------------------
def test_physical_cores_not_multiprocessing():
    s = read("microbench/cpu_bench.py")
    assert "logical=False" in s, "physical_cores must use psutil logical=False"
    assert "multiprocessing.cpu_count()" not in s, "must not report logical count as physical"

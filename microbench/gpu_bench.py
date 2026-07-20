#!/usr/bin/env python3
"""Portable GPU hardware benchmark: raw compute (TFLOPS), memory bandwidth,
and ResNet-50 fp16 inference throughput (eager / torch.compile / TensorRT).
Runs on any CUDA GPU. Usage: python gpu_bench.py"""
import time, json, os, platform, sys

import torch
# Blackwell sm_120 guard (harmless elsewhere)
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

assert torch.cuda.is_available(), "no CUDA GPU"
dev = torch.device("cuda:0")
def sync(): torch.cuda.synchronize()

import time as _time
REPEATS = int(os.environ.get("REPEATS", "3"))
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
try:
    _driver = torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None
except Exception:
    _driver = None
res = {"gpu": name, "capability": f"sm_{cap[0]}{cap[1]}", "vram_GB": round(props.total_memory/1e9, 1),
       "sms": props.multi_processor_count, "torch": torch.__version__,
       "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": _driver,
       "platform": platform.platform(), "utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
       "repeats": REPEATS}

def _median(vals):
    import statistics
    return statistics.median(vals)

def _stats(vals):   # median headline + spread over REPEATS trials (retain the spread in JSON)
    import statistics
    return {"median": round(statistics.median(vals), 1), "min": round(min(vals), 1), "max": round(max(vals), 1)}

def timed(fn, iters, warmup=10):
    for _ in range(warmup): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    sync(); return (time.perf_counter() - t0) / iters

def matmul_tflops(dtype, tf32=False, n=8192, iters=20):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    dt = timed(lambda: torch.matmul(a, b), iters, warmup=5)
    del a, b; torch.cuda.empty_cache()
    return 2 * n**3 / dt / 1e12

print(f"=== {name} | {res['capability']} | {res['vram_GB']} GB | {res['sms']} SMs")
print(f"    torch {torch.__version__} | CUDA {res['cuda']} | cuDNN {res['cudnn']} | median of {REPEATS} ===")
# Sweep matrix sizes so big GPUs (A100/H200) reach peak; override e.g. MATMUL_SIZES=16384,24576,32768
MATMUL_SIZES = [int(s) for s in os.environ.get("MATMUL_SIZES", "8192,16384").split(",")]
print(f"\n[1] Peak matmul TFLOPS (sweep n={MATMUL_SIZES})")
res["tflops"] = {}       # peak per dtype
res["tflops_curve"] = {} # per-n per dtype
for label, dt, tf32 in [("fp32", torch.float32, False), ("tf32", torch.float32, True),
                        ("fp16", torch.float16, False), ("bf16", torch.bfloat16, False)]:
    best = 0.0; best_stats = None; curve = {}
    for n in MATMUL_SIZES:
        try:
            s = _stats([matmul_tflops(dt, tf32=tf32, n=n) for _ in range(REPEATS)])  # median + min/max
            curve[n] = s["median"]
            if s["median"] > best:
                best = s["median"]; best_stats = s
        except Exception:
            curve[n] = None
        torch.cuda.empty_cache()
    # Retain the spread (min/max), like bandwidth/resnet — not just the median (finding #9).
    res["tflops"][label] = best_stats
    res["tflops_curve"][label] = curve
    if best_stats:
        print(f"   {label:5s}: peak {best_stats['median']:8.1f} TFLOPS   "
              f"(min {best_stats['min']}, max {best_stats['max']}; by n: {curve})")
    else:
        print(f"   {label:5s}: unsupported / failed")

print(f"\n[2] Memory bandwidth (DtoD copy, 1 GB), median of {REPEATS}")
try:
    n = 256 * 1024 * 1024  # fp32 elements = 1 GB
    x = torch.empty(n, device=dev, dtype=torch.float32); y = torch.empty_like(x)
    s = _stats([2 * x.numel() * 4 / timed(lambda: y.copy_(x), 50) / 1e9 for _ in range(REPEATS)])
    res["bandwidth_GBs"] = s; print(f"   {s['median']:8.1f} GB/s  (min {s['min']}, max {s['max']})")
    del x, y; torch.cuda.empty_cache()
except Exception as e:
    res["bandwidth_GBs"] = None; print("   ERR", e)

def resnet_throughput(mode, bs=64, iters=30):
    import torchvision
    m = torchvision.models.resnet50(weights=None).to(dev).eval().to(memory_format=torch.channels_last).half()
    if mode == "compile":
        m = torch.compile(m, mode="reduce-overhead")  # cudagraphs; fast compile per shape
    x = torch.randn(bs, 3, 224, 224, device=dev, dtype=torch.half).to(memory_format=torch.channels_last)
    with torch.no_grad():
        dt = timed(lambda: m(x), iters, warmup=15)
    return bs / dt

def resnet_tensorrt(bs=64, iters=50):
    # TRT 10/11: strongly-typed network, precision inferred from the fp16 ONNX.
    import tensorrt as trt, io, torchvision
    m = torchvision.models.resnet50(weights=None).eval().to(dev).half()
    x = torch.randn(bs, 3, 224, 224, device=dev, dtype=torch.half)
    buf = io.BytesIO()
    torch.onnx.export(m, x, buf, input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    assert trt.OnnxParser(network, logger).parse(buf.getvalue()), "onnx parse failed"
    serialized = builder.build_serialized_network(network, builder.create_builder_config())
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network() returned None")
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT deserialize_cuda_engine() returned None")
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("TensorRT create_execution_context() returned None")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = torch.randn(bs, 3, 224, 224, device=dev, dtype=torch.half).contiguous()
    out = torch.empty(bs, 1000, device=dev, dtype=torch.half).contiguous()
    if not (ctx.set_tensor_address(names[0], inp.data_ptr()) and ctx.set_tensor_address(names[1], out.data_ptr())):
        raise RuntimeError("TensorRT set_tensor_address failed")
    s = torch.cuda.current_stream().cuda_stream
    def step():   # a False return would otherwise let us time a no-op and report bogus img/s
        if not ctx.execute_async_v3(s):
            raise RuntimeError("TensorRT execute_async_v3 failed")
    dt = timed(step, iters, warmup=20)
    return bs / dt

# Sweep batches so big GPUs (A100/H200) aren't understated; report PEAK img/s.
# Override with e.g. BATCHES=128,256,512,1024 to push high-VRAM GPUs harder.
BATCHES = [int(b) for b in os.environ.get("BATCHES", "64,128,256").split(",")]
print(f"\n[3] ResNet-50 fp16 inference throughput (peak over batches {BATCHES})")
res["resnet50_fp16_imgs"] = {}   # peak img/s per tier
res["resnet50_curve"] = {}       # img/s per batch per tier
for mode in ["eager", "compile", "tensorrt"]:
    best = 0.0; best_stats = None; curve = {}
    for bs in BATCHES:
        try:
            fn = (lambda: resnet_tensorrt(bs=bs)) if mode == "tensorrt" else (lambda: resnet_throughput(mode, bs=bs))
            s = _stats([fn() for _ in range(REPEATS)])    # median + min/max over REPEATS warm trials
            curve[bs] = round(s["median"])
            if s["median"] > best:
                best = s["median"]; best_stats = s
        except Exception as e:
            curve[bs] = None
            if bs == BATCHES[0]:  # first batch failed -> tier unsupported
                print(f"   {mode:9s}: skipped ({str(e)[:70]})")
        torch.cuda.empty_cache()
    res["resnet50_fp16_imgs"][mode] = best_stats   # {median,min,max} of the peak batch (or None)
    res["resnet50_curve"][mode] = curve
    if best_stats:
        print(f"   {mode:9s}: peak {best_stats['median']:8.0f} img/s  "
              f"(min {best_stats['min']}, max {best_stats['max']}; by batch: {curve})")

print("\n===== JSON =====")
print(json.dumps(res, indent=2))

# Fail loudly if NOTHING was measured — otherwise a GPU that errored on every tier still exits 0 and
# the run-bundle wrapper marks the run PASS with an all-null result (finding #9).
_measured = (any(res["tflops"].values()) or res.get("bandwidth_GBs")
             or any(res["resnet50_fp16_imgs"].values()))
if not _measured:
    print("!! every GPU metric failed — no measurement produced; exiting non-zero", file=sys.stderr)
    sys.exit(1)

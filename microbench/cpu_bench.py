#!/usr/bin/env python3
"""Portable CPU hardware benchmark: matmul GFLOPS (fp32/bf16), memory bandwidth,
and ResNet-50 inference throughput (eager / bf16-autocast / torch.compile).
CPU-only, runs anywhere. Usage: python cpu_bench.py   (REPEATS=5 for more trials)

Headline numbers are the MEDIAN of REPEATS warm trials (min/max also recorded), so a
single noisy run doesn't skew a cross-machine comparison. Not a controlled benchmark —
report the full JSON (incl. metadata) when comparing machines."""
import time, json, os, platform, statistics
import torch

REPEATS = int(os.environ.get("REPEATS", "3"))
torch.set_num_threads(os.cpu_count())

def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n: return n
    except Exception:
        pass
    return None   # unknown — do NOT report logical count as physical

def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()

res = {"cpu": cpu_model(), "logical_cores": os.cpu_count(),
       "physical_cores": physical_cores(), "torch_threads": torch.get_num_threads(),
       "torch": torch.__version__, "platform": platform.platform(),
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repeats": REPEATS}

def timed(fn, iters, warmup=3):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters

def stats(vals):   # median headline + spread over REPEATS trials
    return {"median": round(statistics.median(vals), 1),
            "min": round(min(vals), 1), "max": round(max(vals), 1)}

def matmul_gflops(dtype, n=4096, iters=10):
    a = torch.randn(n, n, dtype=dtype); b = torch.randn(n, n, dtype=dtype)
    dt = timed(lambda: torch.matmul(a, b), iters)
    return 2 * n**3 / dt / 1e9  # GFLOPS

_phys = res['physical_cores']   # None when psutil is absent — show '?' in the banner, keep JSON null
print(f"=== {res['cpu']} | {res['logical_cores']} logical / {_phys if _phys is not None else '?'} physical cores "
      f"| torch {torch.__version__} | median of {REPEATS} ===")
print(f"\n[1] Peak matmul GFLOPS (4096^3), median of {REPEATS}")
res["gflops"] = {}
for label, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    try:
        s = stats([matmul_gflops(dt) for _ in range(REPEATS)]); res["gflops"][label] = s
        print(f"   {label:5s}: {s['median']:8.1f} GFLOPS  (min {s['min']}, max {s['max']})")
    except Exception as e:
        res["gflops"][label] = None; print(f"   {label:5s}: ERR {str(e)[:70]}")

print(f"\n[2] Memory bandwidth (copy, 1 GB), median of {REPEATS}")
try:
    n = 256 * 1024 * 1024
    x = torch.empty(n, dtype=torch.float32); y = torch.empty_like(x)
    s = stats([2 * x.numel() * 4 / timed(lambda: y.copy_(x), 20) / 1e9 for _ in range(REPEATS)])
    res["bandwidth_GBs"] = s
    print(f"   {s['median']:8.1f} GB/s  (min {s['min']}, max {s['max']})")
except Exception as e:
    res["bandwidth_GBs"] = None; print("   ERR", e)

def resnet_throughput(mode, bs=8, iters=10):
    import torchvision
    m = torchvision.models.resnet50(weights=None).eval()
    x = torch.randn(bs, 3, 224, 224)
    if mode == "compile":
        m = torch.compile(m)
    with torch.no_grad():
        if mode == "bf16":
            def run():
                with torch.autocast("cpu", dtype=torch.bfloat16): m(x)
        else:
            def run(): m(x)
        dt = timed(run, iters, warmup=2)
    return bs / dt

print(f"\n[3] ResNet-50 inference throughput (batch=8), median of {REPEATS}")
res["resnet50_imgs"] = {}
for mode in ["fp32", "bf16", "compile"]:
    try:
        s = stats([resnet_throughput(mode) for _ in range(REPEATS)]); res["resnet50_imgs"][mode] = s
        print(f"   {mode:8s}: {s['median']:8.1f} img/s  (min {s['min']}, max {s['max']})")
    except Exception as e:
        res["resnet50_imgs"][mode] = None; print(f"   {mode:8s}: skipped ({str(e)[:70]})")

print("\n===== JSON =====")
print(json.dumps(res, indent=2))

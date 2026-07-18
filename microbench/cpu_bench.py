#!/usr/bin/env python3
"""Portable CPU hardware benchmark: matmul GFLOPS (fp32/bf16), memory bandwidth,
and ResNet-50 inference throughput (eager / bf16-autocast / torch.compile).
CPU-only, runs anywhere. Usage: python cpu_bench.py"""
import time, json, os, platform, multiprocessing
import torch

torch.set_num_threads(os.cpu_count())

def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()

res = {"cpu": cpu_model(), "logical_cores": os.cpu_count(),
       "physical_cores": multiprocessing.cpu_count(), "torch_threads": torch.get_num_threads(),
       "torch": torch.__version__}

def timed(fn, iters, warmup=3):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t0) / iters

def matmul_gflops(dtype, n=4096, iters=10):
    a = torch.randn(n, n, dtype=dtype); b = torch.randn(n, n, dtype=dtype)
    dt = timed(lambda: torch.matmul(a, b), iters)
    return 2 * n**3 / dt / 1e9  # GFLOPS

print(f"=== {res['cpu']} | {res['logical_cores']} threads | torch {torch.__version__} ===")
print("\n[1] Peak matmul GFLOPS (4096^3)")
res["gflops"] = {}
for label, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    try:
        v = matmul_gflops(dt); res["gflops"][label] = round(v, 1)
        print(f"   {label:5s}: {v:8.1f} GFLOPS")
    except Exception as e:
        res["gflops"][label] = None; print(f"   {label:5s}: ERR {str(e)[:70]}")

print("\n[2] Memory bandwidth (copy, 1 GB)")
try:
    n = 256 * 1024 * 1024
    x = torch.empty(n, dtype=torch.float32); y = torch.empty_like(x)
    dt = timed(lambda: y.copy_(x), 20)
    res["bandwidth_GBs"] = round(2 * x.numel() * 4 / dt / 1e9, 1)
    print(f"   {res['bandwidth_GBs']:8.1f} GB/s")
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

print("\n[3] ResNet-50 inference throughput (batch=8)")
res["resnet50_imgs"] = {}
for mode in ["fp32", "bf16", "compile"]:
    try:
        v = resnet_throughput(mode); res["resnet50_imgs"][mode] = round(v, 1)
        print(f"   {mode:8s}: {v:8.1f} img/s")
    except Exception as e:
        res["resnet50_imgs"][mode] = None; print(f"   {mode:8s}: skipped ({str(e)[:70]})")

print("\n===== JSON =====")
print(json.dumps(res, indent=2))

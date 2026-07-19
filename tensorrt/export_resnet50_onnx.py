#!/usr/bin/env python3
"""Export a torchvision ResNet-50 to an fp16, dynamic-batch ONNX.

Usage: python export_resnet50_onnx.py <out.onnx>

Weights: uses $RESNET50_PTH if it points at an existing checkpoint (e.g. the
MLPerf/Zenodo `resnet50-19c8e357.pth`); otherwise torchvision downloads the
identical IMAGENET1K_V1 weights automatically. So this runs on a fresh machine
with no pre-staged files.
"""
import os, sys, torch, torchvision

out = sys.argv[1] if len(sys.argv) > 1 else "resnet50_fp16_dyn.onnx"
pth = os.environ.get("RESNET50_PTH", "")

if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA GPU visible — this exports an fp16 model built on cuda. "
             "Run it on the GPU box (check CUDA_VISIBLE_DEVICES if a GPU is present).")

if pth and os.path.exists(pth):
    try:
        sd = torch.load(pth, map_location="cpu", weights_only=True)   # safe path first
    except Exception as e:
        if os.environ.get("ALLOW_UNSAFE_PICKLE") == "1":
            print(f"[warn] safe load failed ({e}); ALLOW_UNSAFE_PICKLE=1 → weights_only=False "
                  f"(arbitrary-code-execution risk — only for files you trust)")
            sd = torch.load(pth, map_location="cpu", weights_only=False)
        else:
            sys.exit(f"ERROR: {pth} could not be loaded safely (weights_only=True): {e}\n"
                     f"Legacy checkpoints need weights_only=False, which can execute arbitrary code. "
                     f"Re-run with ALLOW_UNSAFE_PICKLE=1 ONLY if you trust this file, or unset "
                     f"RESNET50_PTH to download verified torchvision weights instead.")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    m = torchvision.models.resnet50()
    m.load_state_dict(sd)
    print(f"loaded weights from {pth}")
else:
    if pth:
        print(f"[info] $RESNET50_PTH={pth} not found; downloading torchvision IMAGENET1K_V1 weights")
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

m.eval().cuda().half()
x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.half)
torch.onnx.export(m, x, out, input_names=["x"], output_names=["y"], opset_version=17,
                  dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, dynamo=False)
print("wrote", out)

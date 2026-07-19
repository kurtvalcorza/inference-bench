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

if pth and os.path.exists(pth):
    sd = torch.load(pth, map_location="cpu", weights_only=False)
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

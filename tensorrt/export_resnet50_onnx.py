import sys, torch, torchvision
out = sys.argv[1]
sd = torch.load("/root/mlperf/vision/resnet50-19c8e357.pth", map_location="cpu", weights_only=False)
m = torchvision.models.resnet50(); m.load_state_dict(sd); m.eval().cuda().half()
x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.half)
torch.onnx.export(m, x, out, input_names=["x"], output_names=["y"], opset_version=17,
                  dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, dynamo=False)
print("wrote", out)

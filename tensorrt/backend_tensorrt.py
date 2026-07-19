"""
TensorRT backend for the MLPerf vision harness.
--model must point at an fp16 ResNet-50 ONNX with a dynamic batch axis
(build one with export_resnet50_onnx.py). Builds a dynamic-batch fp16 engine
(1..MAX_BATCHSIZE) and runs inference via torch CUDA tensors as I/O buffers.
"""
# pylint: disable=missing-docstring
import os
import threading
import numpy as np
import torch
import tensorrt as trt
import backend

MAXBS = int(os.environ.get("TRT_MAX_BATCHSIZE", "32"))


class BackendTensorRT(backend.Backend):
    def __init__(self):
        super().__init__()
        self.dev = torch.device("cuda:0")
        self.logger = trt.Logger(trt.Logger.ERROR)
        # TRT execution contexts are NOT thread-safe; MLPerf's QueueRunner calls
        # predict() from many worker threads, so serialize GPU access.
        self.lock = threading.Lock()

    def version(self):
        return trt.__version__

    def name(self):
        return "tensorrt"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        # parse the fp16 ONNX (strongly-typed => fp16 engine), dynamic batch profile
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self.logger)
        with open(model_path, "rb") as f:
            assert parser.parse(f.read()), "ONNX parse failed: " + \
                str([parser.get_error(i).desc() for i in range(parser.num_errors)])
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        inp = network.get_input(0)
        c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
        profile.set_shape(inp.name, (1, c, h, w), (MAXBS, c, h, w), (MAXBS, c, h, w))
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:                      # NVIDIA: build can fail and return None
            raise RuntimeError("TensorRT build_serialized_network() returned None (engine build failed)")
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError("TensorRT deserialize_cuda_engine() returned None")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT create_execution_context() returned None")
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        out_shape = self.engine.get_tensor_shape(self.out_name)   # (-1, 1000)
        self.n_classes = int(out_shape[1])
        self.chw = (c, h, w)
        # reusable max-size device buffers
        self.d_in = torch.empty((MAXBS, c, h, w), device=self.dev, dtype=torch.half)
        self.d_out = torch.empty((MAXBS, self.n_classes), device=self.dev, dtype=torch.half)
        self.stream = torch.cuda.current_stream().cuda_stream
        if not self.context.set_tensor_address(self.in_name, self.d_in.data_ptr()):
            raise RuntimeError("TensorRT set_tensor_address(input) failed")
        if not self.context.set_tensor_address(self.out_name, self.d_out.data_ptr()):
            raise RuntimeError("TensorRT set_tensor_address(output) failed")
        self.inputs = inputs if inputs else [self.in_name]
        self.outputs = outputs if outputs else [self.out_name]
        # warm up (build lazy kernels / init) so the first real query isn't a huge outlier
        for bs in (1, MAXBS):
            if not self.context.set_input_shape(self.in_name, (bs,) + self.chw):
                raise RuntimeError(f"TensorRT set_input_shape failed for batch {bs}")
            if not self.context.execute_async_v3(self.stream):
                raise RuntimeError(f"TensorRT execute_async_v3 failed during warmup (batch {bs})")
        torch.cuda.synchronize()
        return self

    def predict(self, feed):
        data = feed[list(feed.keys())[0]]
        bs = data.shape[0]
        with self.lock:
            x = torch.as_tensor(np.ascontiguousarray(data), dtype=torch.half, device=self.dev)
            self.d_in[:bs].copy_(x)
            if not self.context.set_input_shape(self.in_name, (bs,) + self.chw):
                raise RuntimeError(f"TensorRT set_input_shape failed for batch {bs}")
            if not self.context.execute_async_v3(self.stream):   # else we'd time a no-op and report bogus throughput
                raise RuntimeError(f"TensorRT execute_async_v3 failed for batch {bs}")
            torch.cuda.synchronize()
            return [self.d_out[:bs].float().cpu().numpy()]

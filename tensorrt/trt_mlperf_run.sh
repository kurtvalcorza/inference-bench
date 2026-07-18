#!/usr/bin/env bash
set -uo pipefail
source /root/mlperf/venv/bin/activate
SP="/mnt/c/Users/KURTVA~1/AppData/Local/Temp/claude/C--Users-Kurt-Valcorza-Projects/50125cc2-9014-4c49-8f75-3526787d224f/scratchpad"
H=/root/mlperf/inference/vision/classification_and_detection
DATA=/root/mlperf/vision/inet_val          # representative ImageNet-1k subset (5/class, 1000 classes)
VMAP=$DATA/val_map.txt
ONNX=/root/mlperf/vision/resnet50_fp16_dyn.onnx
MAXBS=${MAXBS:-128}
export TRT_MAX_BATCHSIZE=$MAXBS
CONF=/root/mlperf/vision/trt.conf
cat > "$CONF" <<'CONF'
resnet50.SingleStream.min_duration = 10000
resnet50.SingleStream.min_query_count = 4000
resnet50.Offline.target_qps = 12000
resnet50.Offline.min_duration = 10000
resnet50.Offline.min_query_count = 1
CONF

echo "===== install TensorRT backend into the harness ====="
cp "$SP/backend_tensorrt.py" "$H/python/backend_tensorrt.py"
python - <<'PY'
f="/root/mlperf/inference/vision/classification_and_detection/python/main.py"; s=open(f).read()
if "backend_tensorrt" not in s:
    a='    elif backend == "pytorch-native":'
    ins='    elif backend == "tensorrt":\n        from backend_tensorrt import BackendTensorRT\n\n        backend = BackendTensorRT()\n'
    s=s.replace(a, ins+a, 1); open(f,"w").write(s); print("main.py patched: +tensorrt backend")
else: print("main.py already has tensorrt backend")
PY

echo
echo "===== export fp16 dynamic-batch ONNX ====="
[ -s "$ONNX" ] || python "$SP/export_resnet50_onnx.py" "$ONNX"
ls -lh "$ONNX"

cd "$H/python"
run () {  # scenario, extra-args, outdir
  python main.py --profile resnet50-pytorch --backend tensorrt --model "$ONNX" \
    --dataset-path "$DATA" --user_conf "$CONF" --max-batchsize $MAXBS \
    --scenario "$1" $2 --output "$3" >/tmp/trt_$1.log 2>&1 || true
}

echo
echo "############ SingleStream (p50/p90/p99 latency) ############"
run SingleStream "" /root/mlperf/vision/trt_ss
sed -n '1,22p' /root/mlperf/vision/trt_ss/mlperf_log_summary.txt 2>/dev/null || { echo "(no summary)"; tail -15 /tmp/trt_SingleStream.log; }

echo
echo "############ Offline (throughput) ############"
run Offline "" /root/mlperf/vision/trt_off
sed -n '1,14p' /root/mlperf/vision/trt_off/mlperf_log_summary.txt 2>/dev/null || echo "(no summary)"

echo
echo "############ Accuracy (top-1, fp16 TRT) ############"
run Offline "--accuracy" /root/mlperf/vision/trt_acc
python ../tools/accuracy-imagenet.py --mlperf-accuracy-file /root/mlperf/vision/trt_acc/mlperf_log_accuracy.json \
  --imagenet-val-file "$VMAP" --dtype float32 2>&1 | tail -1
echo "DONE trt mlperf"

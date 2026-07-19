# Verified asset checksums

SHA-256 hashes of the models/datasets the reference notebooks download, computed from the copies
used to produce the results in this repo. The hardened notebook cells verify each download against
these before use (and refuse to proceed / extract on a mismatch). The harness itself is pinned to
`mlcommons/inference` commit **`da738a5`** (`INFERENCE_REF`).

| Asset | Source | SHA-256 |
|---|---|---|
| `resnet50-19c8e357.pth` (ResNet-50 weights) | Zenodo record 4588417 | `19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097` |
| `imagenette2-320.tgz` (fast.ai Imagenette) | fast-ai-imageclas S3 | **unpinned** (see note) |
| `model.pytorch` (BERT-Large) | Zenodo record 3733896 | `71af14acc3cb47ebd88e028d9ff8a5f06e15f6ed666e16293b7ad2539171397f` |
| `vocab.txt` (BERT) | Zenodo record 3733896 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| `dev-v1.1.json` (SQuAD v1.1 dev) | rajpurkar/SQuAD-explorer | `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` |
| `dev-clean.tar.gz` (LibriSpeech) | OpenSLR resource 12 | `76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3` |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | TheBloke HF (GGUF) | `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` |

Notes:
- **`imagenette2-320.tgz` is not yet pinned.** Re-hashing the archive requires re-downloading it, and
  fast.ai's S3 was throttling to ~1.6 KB/s this session (the ~325 MB download would not complete). The
  resnet notebook and `trt_mlperf_run.sh` therefore run it in **record mode** (download → print hash →
  extract) and enforce only when `IMAGENETTE_SHA256` is set. To pin it: run once on a good connection,
  copy the printed hash here and into your env (`IMAGENETTE_SHA256=<hash>`). Every *other* asset above
  is hash-enforced. The representative 1000-class subset is the checksum-independent alternative
  (built + validated by `tensorrt/build_imagenet_subset.py`).
- **whisper-large-v3** weights are fetched by `openai-whisper` (`whisper.load_model('large-v3')`),
  which verifies its own built-in per-model SHA-256 — no extra check needed.
- **torchvision ResNet-50** downloaded via `torchvision.models.resnet50(weights=...)` is hash-checked
  by torchvision (the URL embeds the hash). The exporter's `RESNET50_PTH` path (Zenodo `.pth` above)
  loads with `weights_only=True` first, falling back to `weights_only=False` only for the verified,
  legacy tar-format checkpoint (opt-in via `ALLOW_UNSAFE_PICKLE=1` in `export_resnet50_onnx.py`).

# Colab Session: mlperf

## Session Created: 2026-07-18 11:06:17
- Endpoint: `gpu-t4-s-kkb-usw1b2-24ofesrmbxhn5`

### Execution (2026-07-18 11:07:32)
```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
else:
    print("!! No GPU — set Runtime -> Change runtime type -> GPU")
```

**Output**:
```
name, memory.total [MiB], driver_version
Tesla T4, 15360 MiB, 580.82.07
```

**Output**:
```
torch 2.11.0+cu128 | CUDA available: True
device: Tesla T4 | capability: (7, 5)
```

### Execution (2026-07-18 11:07:46)
```python
!pip -q install mlcommons-loadgen "transformers==4.48.3"
import mlperf_loadgen, transformers
print("loadgen + transformers", transformers.__version__, "ready")
```

**Output**:
```
[?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/44.4 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.4/44.4 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
[?25h```

**Output**:
```
[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/9.7 MB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━━━━━[0m[90m╺[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/9.7 MB[0m [31m65.3 MB/s[0m eta [36m0:00:01[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m9.7/9.7 MB[0m [31m147.9 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.7/9.7 MB[0m [31m94.7 MB/s[0m eta [36m0:00:00[0m
[?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/472.2 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m472.2/472.2 kB[0m [31m46.0 MB/s[0m eta [36m0:00:00[0m
[?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/566.4 kB[0m [31m?[0m eta [36m-:--:--[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m566.4/566.4 kB[0m [31m51.5 MB/s[0m eta [36m0:00:00[0m
[?25h[?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/3.1 MB[0m [31m?[0m eta [36m-:--:--[0m```

**Output**:
```
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m117.7 MB/s[0m eta [36m0:00:00[0m
[?25h```

**Output**:
```
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio 6.20.0 requires huggingface-hub<2.0,>=1.2.0, but you have huggingface-hub 0.36.2 which is incompatible.[0m[31m
[0m```

**Output**:
```
loadgen + transformers 4.48.3 ready
```

### Execution (2026-07-18 11:08:04)
```python
import os
if not os.path.isdir('/content/inference'):
    !git clone --depth 1 https://github.com/mlcommons/inference.git /content/inference
%cd /content/inference/language/bert
!git -C /content/inference rev-parse --short HEAD
```

**Output**:
```
Cloning into '/content/inference'...
```

**Output**:
```
remote: Enumerating objects: 1467, done.[K
remote: Counting objects:   0% (1/1467)[Kremote: Counting objects:   1% (15/1467)[Kremote: Counting objects:   2% (30/1467)[Kremote: Counting objects:   3% (45/1467)[Kremote: Counting objects:   4% (59/1467)[Kremote: Counting objects:   5% (74/1467)[Kremote: Counting objects:   6% (89/1467)[Kremote: Counting objects:   7% (103/1467)[Kremote: Counting objects:   8% (118/1467)[Kremote: Counting objects:   9% (133/1467)[Kremote: Counting objects:  10% (147/1467)[Kremote: Counting objects:  11% (162/1467)[Kremote: Counting objects:  12% (177/1467)[Kremote: Counting objects:  13% (191/1467)[Kremote: Counting objects:  14% (206/1467)[Kremote: Counting objects:  15% (221/1467)[Kremote: Counting objects:  16% (235/1467)[Kremote: Counting objects:  17% (250/1467)[Kremote: Counting objects:  18% (265/1467)[Kremote: Counting objects:  19% (279/1467)[Kremote: Counting objects:  20% (294/1467)[Kremote: Counting objects:  21% (309/1467)[Kremote: Counting objects:  22% (323/1467)[Kremote: Counting objects:  23% (338/1467)[Kremote: Counting objects:  24% (353/1467)[Kremote: Counting objects:  25% (367/1467)[Kremote: Counting objects:  26% (382/1467)[Kremote: Counting objects:  27% (397/1467)[Kremote: Counting objects:  28% (411/1467)[Kremote: Counting objects:  29% (426/1467)[Kremote: Counting objects:  30% (441/1467)[Kremote: Counting objects:  31% (455/1467)[Kremote: Counting objects:  32% (470/1467)[Kremote: Counting objects:  33% (485/1467)[Kremote: Counting objects:  34% (499/1467)[Kremote: Counting objects:  35% (514/1467)[Kremote: Counting objects:  36% (529/1467)[Kremote: Counting objects:  37% (543/1467)[Kremote: Counting objects:  38% (558/1467)[Kremote: Counting objects:  39% (573/1467)[Kremote: Counting objects:  40% (587/1467)[Kremote: Counting objects:  41% (602/1467)[Kremote: Counting objects:  42% (617/1467)[Kremote: Counting objects:  43% (631/1467)[Kremote: Counting objects:  44% (646/1467)[Kremote: Counting objects:  45% (661/1467)[Kremote: Counting objects:  46% (675/1467)[Kremote: Counting objects:  47% (690/1467)[Kremote: Counting objects:  48% (705/1467)[Kremote: Counting objects:  49% (719/1467)[Kremote: Counting objects:  50% (734/1467)[Kremote: Counting objects:  51% (749/1467)[Kremote: Counting objects:  52% (763/1467)[Kremote: Counting objects:  53% (778/1467)[Kremote: Counting objects:  54% (793/1467)[Kremote: Counting objects:  55% (807/1467)[Kremote: Counting objects:  56% (822/1467)[Kremote: Counting objects:  57% (837/1467)[Kremote: Counting objects:  58% (851/1467)[Kremote: Counting objects:  59% (866/1467)[Kremote: Counting objects:  60% (881/1467)[Kremote: Counting objects:  61% (895/1467)[Kremote: Counting objects:  62% (910/1467)[Kremote: Counting objects:  63% (925/1467)[Kremote: Counting objects:  64% (939/1467)[Kremote: Counting objects:  65% (954/1467)[Kremote: Counting objects:  66% (969/1467)[Kremote: Counting objects:  67% (983/1467)[Kremote: Counting objects:  68% (998/1467)[Kremote: Counting objects:  69% (1013/1467)[Kremote: Counting objects:  70% (1027/1467)[Kremote: Counting objects:  71% (1042/1467)[Kremote: Counting objects:  72% (1057/1467)[Kremote: Counting objects:  73% (1071/1467)[Kremote: Counting objects:  74% (1086/1467)[Kremote: Counting objects:  75% (1101/1467)[Kremote: Counting objects:  76% (1115/1467)[Kremote: Counting objects:  77% (1130/1467)[Kremote: Counting objects:  78% (1145/1467)[Kremote: Counting objects:  79% (1159/1467)[Kremote: Counting objects:  80% (1174/1467)[Kremote: Counting objects:  81% (1189/1467)[Kremote: Counting objects:  82% (1203/1467)[Kremote: Counting objects:  83% (1218/1467)[Kremote: Counting objects:  84% (1233/1467)[Kremote: Counting objects:  85% (1247/1467)[Kremote: Counting objects:  86% (1262/1467)[Kremote: Counting objects:  87% (1277/1467)[Kremote: Counting objects:  88% (1291/1467)[Kremote: Counting objects:  89% (1306/1467)[Kremote: Counting objects:  90% (1321/1467)[Kremote: Counting objects:  91% (1335/1467)[Kremote: Counting objects:  92% (1350/1467)[Kremote: Counting objects:  93% (1365/1467)[Kremote: Counting objects:  94% (1379/1467)[Kremote: Counting objects:  95% (1394/1467)[Kremote: Counting objects:  96% (1409/1467)[Kremote: Counting objects:  97% (1423/1467)[Kremote: Counting objects:  98% (1438/1467)[Kremote: Counting objects:  99% (1453/1467)[Kremote: Counting objects: 100% (1467/1467)[Kremote: Counting objects: 100% (1467/1467), done.[K
remote: Compressing objects:   0% (1/1120)[Kremote: Compressing objects:   1% (12/1120)[Kremote: Compressing objects:   2% (23/1120)[Kremote: Compressing objects:   3% (34/1120)[Kremote: Compressing objects:   4% (45/1120)[Kremote: Compressing objects:   5% (56/1120)[Kremote: Compressing objects:   6% (68/1120)[K```

**Output**:
```
remote: Compressing objects:   7% (79/1120)[Kremote: Compressing objects:   8% (90/1120)[Kremote: Compressing objects:   9% (101/1120)[Kremote: Compressing objects:  10% (112/1120)[Kremote: Compressing objects:  11% (124/1120)[Kremote: Compressing objects:  12% (135/1120)[Kremote: Compressing objects:  13% (146/1120)[Kremote: Compressing objects:  14% (157/1120)[Kremote: Compressing objects:  15% (168/1120)[Kremote: Compressing objects:  16% (180/1120)[Kremote: Compressing objects:  17% (191/1120)[Kremote: Compressing objects:  18% (202/1120)[Kremote: Compressing objects:  19% (213/1120)[Kremote: Compressing objects:  20% (224/1120)[Kremote: Compressing objects:  21% (236/1120)[Kremote: Compressing objects:  22% (247/1120)[Kremote: Compressing objects:  23% (258/1120)[Kremote: Compressing objects:  24% (269/1120)[Kremote: Compressing objects:  25% (280/1120)[Kremote: Compressing objects:  26% (292/1120)[Kremote: Compressing objects:  27% (303/1120)[Kremote: Compressing objects:  28% (314/1120)[Kremote: Compressing objects:  29% (325/1120)[Kremote: Compressing objects:  30% (336/1120)[Kremote: Compressing objects:  31% (348/1120)[Kremote: Compressing objects:  32% (359/1120)[Kremote: Compressing objects:  33% (370/1120)[Kremote: Compressing objects:  34% (381/1120)[Kremote: Compressing objects:  35% (392/1120)[Kremote: Compressing objects:  36% (404/1120)[Kremote: Compressing objects:  37% (415/1120)[Kremote: Compressing objects:  38% (426/1120)[Kremote: Compressing objects:  39% (437/1120)[Kremote: Compressing objects:  40% (448/1120)[Kremote: Compressing objects:  41% (460/1120)[Kremote: Compressing objects:  42% (471/1120)[Kremote: Compressing objects:  43% (482/1120)[Kremote: Compressing objects:  44% (493/1120)[Kremote: Compressing objects:  45% (504/1120)[K```

**Output**:
```
remote: Compressing objects:  45% (512/1120)[K```

**Output**:
```
remote: Compressing objects:  45% (513/1120)[K```

**Output**:
```
remote: Compressing objects:  45% (514/1120)[K```

**Output**:
```
remote: Compressing objects:  45% (515/1120)[Kremote: Compressing objects:  46% (516/1120)[Kremote: Compressing objects:  47% (527/1120)[Kremote: Compressing objects:  48% (538/1120)[Kremote: Compressing objects:  49% (549/1120)[Kremote: Compressing objects:  50% (560/1120)[Kremote: Compressing objects:  51% (572/1120)[Kremote: Compressing objects:  52% (583/1120)[Kremote: Compressing objects:  53% (594/1120)[Kremote: Compressing objects:  54% (605/1120)[Kremote: Compressing objects:  55% (616/1120)[Kremote: Compressing objects:  56% (628/1120)[Kremote: Compressing objects:  57% (639/1120)[Kremote: Compressing objects:  58% (650/1120)[Kremote: Compressing objects:  59% (661/1120)[Kremote: Compressing objects:  60% (672/1120)[Kremote: Compressing objects:  61% (684/1120)[Kremote: Compressing objects:  62% (695/1120)[Kremote: Compressing objects:  63% (706/1120)[Kremote: Compressing objects:  64% (717/1120)[Kremote: Compressing objects:  65% (728/1120)[Kremote: Compressing objects:  66% (740/1120)[Kremote: Compressing objects:  67% (751/1120)[Kremote: Compressing objects:  68% (762/1120)[Kremote: Compressing objects:  69% (773/1120)[Kremote: Compressing objects:  70% (784/1120)[Kremote: Compressing objects:  71% (796/1120)[Kremote: Compressing objects:  72% (807/1120)[Kremote: Compressing objects:  73% (818/1120)[Kremote: Compressing objects:  74% (829/1120)[Kremote: Compressing objects:  75% (840/1120)[Kremote: Compressing objects:  76% (852/1120)[Kremote: Compressing objects:  77% (863/1120)[Kremote: Compressing objects:  78% (874/1120)[Kremote: Compressing objects:  79% (885/1120)[Kremote: Compressing objects:  80% (896/1120)[Kremote: Compressing objects:  81% (908/1120)[Kremote: Compressing objects:  82% (919/1120)[Kremote: Compressing objects:  83% (930/1120)[Kremote: Compressing objects:  84% (941/1120)[Kremote: Compressing objects:  85% (952/1120)[Kremote: Compressing objects:  86% (964/1120)[Kremote: Compressing objects:  87% (975/1120)[Kremote: Compressing objects:  88% (986/1120)[Kremote: Compressing objects:  89% (997/1120)[Kremote: Compressing objects:  90% (1008/1120)[Kremote: Compressing objects:  91% (1020/1120)[Kremote: Compressing objects:  92% (1031/1120)[Kremote: Compressing objects:  93% (1042/1120)[Kremote: Compressing objects:  94% (1053/1120)[Kremote: Compressing objects:  95% (1064/1120)[Kremote: Compressing objects:  96% (1076/1120)[Kremote: Compressing objects:  97% (1087/1120)[Kremote: Compressing objects:  98% (1098/1120)[Kremote: Compressing objects:  99% (1109/1120)[Kremote: Compressing objects: 100% (1120/1120)[Kremote: Compressing objects: 100% (1120/1120), done.[K
Receiving objects:   0% (1/1467)Receiving objects:   1% (15/1467)Receiving objects:   2% (30/1467)Receiving objects:   3% (45/1467)Receiving objects:   4% (59/1467)Receiving objects:   5% (74/1467)Receiving objects:   6% (89/1467)Receiving objects:   7% (103/1467)Receiving objects:   8% (118/1467)Receiving objects:   9% (133/1467)Receiving objects:  10% (147/1467)Receiving objects:  11% (162/1467)Receiving objects:  12% (177/1467)Receiving objects:  13% (191/1467)Receiving objects:  14% (206/1467)Receiving objects:  15% (221/1467)Receiving objects:  16% (235/1467)Receiving objects:  17% (250/1467)```

**Output**:
```
Receiving objects:  18% (265/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  19% (279/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  20% (294/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  21% (309/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  22% (323/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  23% (338/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  24% (353/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  25% (367/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  26% (382/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  27% (397/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  28% (411/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  29% (426/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  30% (441/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  31% (455/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  32% (470/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  33% (485/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  34% (499/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  35% (514/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  36% (529/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  37% (543/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  38% (558/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  39% (573/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  40% (587/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  41% (602/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  42% (617/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  43% (631/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  44% (646/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  45% (661/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  46% (675/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  47% (690/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  48% (705/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  49% (719/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  50% (734/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  51% (749/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  52% (763/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  53% (778/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  54% (793/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  55% (807/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  56% (822/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  57% (837/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  58% (851/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  59% (866/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  60% (881/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  61% (895/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  62% (910/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  63% (925/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  64% (939/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  65% (954/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  66% (969/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  67% (983/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  68% (998/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  69% (1013/1467), 4.81 MiB | 9.59 MiB/sReceiving objects:  70% (1027/1467), 4.81 MiB | 9.59 MiB/s```

**Output**:
```
Receiving objects:  70% (1029/1467), 23.25 MiB | 23.22 MiB/s```

**Output**:
```
Receiving objects:  71% (1042/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  72% (1057/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  73% (1071/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  74% (1086/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  75% (1101/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  76% (1115/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  77% (1130/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  78% (1145/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  79% (1159/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  80% (1174/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  81% (1189/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  82% (1203/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  83% (1218/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  84% (1233/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  85% (1247/1467), 42.58 MiB | 28.36 MiB/sReceiving objects:  86% (1262/1467), 42.58 MiB | 28.36 MiB/s```

**Output**:
```
Receiving objects:  86% (1269/1467), 68.11 MiB | 34.04 MiB/s```

**Output**:
```
Receiving objects:  86% (1271/1467), 115.59 MiB | 38.13 MiB/s```

**Output**:
```
Receiving objects:  87% (1277/1467), 115.59 MiB | 38.13 MiB/s```

**Output**:
```
Receiving objects:  87% (1281/1467), 160.22 MiB | 39.75 MiB/s```

**Output**:
```
Receiving objects:  88% (1291/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  89% (1306/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  90% (1321/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  91% (1335/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  92% (1350/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  93% (1365/1467), 160.22 MiB | 39.75 MiB/sReceiving objects:  94% (1379/1467), 182.28 MiB | 40.23 MiB/sReceiving objects:  95% (1394/1467), 182.28 MiB | 40.23 MiB/sReceiving objects:  96% (1409/1467), 182.28 MiB | 40.23 MiB/sReceiving objects:  97% (1423/1467), 182.28 MiB | 40.23 MiB/s```

**Output**:
```
Receiving objects:  97% (1435/1467), 191.86 MiB | 41.30 MiB/s```

**Output**:
```
Receiving objects:  98% (1438/1467), 201.80 MiB | 39.42 MiB/sReceiving objects:  98% (1439/1467), 211.61 MiB | 37.32 MiB/s```

**Output**:
```
Receiving objects:  98% (1439/1467), 231.21 MiB | 30.73 MiB/s```

**Output**:
```
Receiving objects:  98% (1443/1467), 250.91 MiB | 25.07 MiB/s```

**Output**:
```
Receiving objects:  99% (1453/1467), 260.59 MiB | 22.30 MiB/sremote: Total 1467 (delta 325), reused 969 (delta 233), pack-reused 0 (from 0)[K
Receiving objects: 100% (1467/1467), 260.59 MiB | 22.30 MiB/sReceiving objects: 100% (1467/1467), 261.35 MiB | 30.52 MiB/s, done.
Resolving deltas:   0% (0/325)Resolving deltas:   1% (4/325)Resolving deltas:   2% (7/325)Resolving deltas:   3% (10/325)Resolving deltas:   4% (13/325)Resolving deltas:   5% (17/325)Resolving deltas:   6% (20/325)Resolving deltas:   7% (23/325)Resolving deltas:   8% (26/325)Resolving deltas:   9% (30/325)Resolving deltas:  10% (33/325)Resolving deltas:  11% (36/325)Resolving deltas:  12% (39/325)Resolving deltas:  13% (43/325)Resolving deltas:  14% (46/325)Resolving deltas:  15% (49/325)Resolving deltas:  16% (52/325)Resolving deltas:  17% (56/325)Resolving deltas:  18% (59/325)Resolving deltas:  19% (62/325)Resolving deltas:  20% (65/325)Resolving deltas:  21% (69/325)Resolving deltas:  22% (72/325)Resolving deltas:  23% (75/325)Resolving deltas:  24% (78/325)Resolving deltas:  25% (82/325)Resolving deltas:  26% (85/325)Resolving deltas:  27% (88/325)Resolving deltas:  28% (91/325)Resolving deltas:  29% (95/325)Resolving deltas:  30% (98/325)Resolving deltas:  31% (101/325)Resolving deltas:  32% (104/325)Resolving deltas:  33% (108/325)Resolving deltas:  34% (111/325)Resolving deltas:  35% (114/325)Resolving deltas:  36% (117/325)Resolving deltas:  37% (121/325)Resolving deltas:  38% (124/325)Resolving deltas:  39% (127/325)Resolving deltas:  40% (130/325)Resolving deltas:  41% (134/325)Resolving deltas:  42% (137/325)Resolving deltas:  43% (140/325)Resolving deltas:  44% (143/325)Resolving deltas:  45% (147/325)Resolving deltas:  46% (150/325)Resolving deltas:  47% (153/325)Resolving deltas:  48% (156/325)Resolving deltas:  49% (160/325)Resolving deltas:  50% (163/325)Resolving deltas:  51% (166/325)Resolving deltas:  52% (169/325)Resolving deltas:  53% (173/325)Resolving deltas:  54% (176/325)Resolving deltas:  55% (179/325)Resolving deltas:  56% (182/325)Resolving deltas:  57% (186/325)Resolving deltas:  58% (189/325)Resolving deltas:  59% (192/325)Resolving deltas:  60% (195/325)Resolving deltas:  61% (199/325)Resolving deltas:  62% (202/325)Resolving deltas:  63% (205/325)Resolving deltas:  64% (208/325)Resolving deltas:  65% (212/325)Resolving deltas:  66% (215/325)Resolving deltas:  67% (218/325)Resolving deltas:  68% (221/325)Resolving deltas:  69% (225/325)Resolving deltas:  70% (228/325)Resolving deltas:  71% (231/325)Resolving deltas:  72% (234/325)Resolving deltas:  73% (238/325)Resolving deltas:  74% (241/325)Resolving deltas:  75% (244/325)Resolving deltas:  76% (247/325)Resolving deltas:  77% (251/325)Resolving deltas:  78% (254/325)Resolving deltas:  79% (257/325)Resolving deltas:  80% (260/325)Resolving deltas:  81% (264/325)Resolving deltas:  82% (267/325)Resolving deltas:  83% (270/325)Resolving deltas:  84% (273/325)Resolving deltas:  85% (277/325)Resolving deltas:  86% (280/325)Resolving deltas:  87% (283/325)Resolving deltas:  88% (286/325)Resolving deltas:  89% (290/325)Resolving deltas:  90% (293/325)Resolving deltas:  91% (296/325)Resolving deltas:  92% (299/325)Resolving deltas:  93% (303/325)Resolving deltas:  94% (306/325)Resolving deltas:  95% (309/325)Resolving deltas:  96% (312/325)Resolving deltas:  97% (316/325)Resolving deltas:  98% (319/325)```

**Output**:
```
Resolving deltas:  99% (322/325)```

**Output**:
```
Resolving deltas:  99% (324/325)Resolving deltas: 100% (325/325)Resolving deltas: 100% (325/325), done.
```

**Output**:
```
Updating files:  87% (1195/1371)Updating files:  88% (1207/1371)Updating files:  89% (1221/1371)Updating files:  90% (1234/1371)Updating files:  91% (1248/1371)Updating files:  92% (1262/1371)Updating files:  93% (1276/1371)Updating files:  94% (1289/1371)Updating files:  95% (1303/1371)Updating files:  96% (1317/1371)Updating files:  97% (1330/1371)```

**Output**:
```
Updating files:  98% (1344/1371)```

**Output**:
```
Updating files:  98% (1346/1371)Updating files:  99% (1358/1371)Updating files: 100% (1371/1371)Updating files: 100% (1371/1371), done.
```

**Output**:
```
/content/inference/language/bert
```

**Output**:
```
da738a5
```

### Execution (2026-07-18 11:12:04)
```python
%%bash
set -e
MODELDIR=build/data/bert_tf_v1_1_large_fp32_384_v2
mkdir -p $MODELDIR build/logs
[ -s build/data/dev-v1.1.json ] || wget -q -O build/data/dev-v1.1.json \
    https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
[ -s $MODELDIR/vocab.txt ] || wget -q -O $MODELDIR/vocab.txt \
    "https://zenodo.org/record/3733896/files/vocab.txt?download=1"
echo "downloading model.pytorch (~1.3 GB) ..."
[ -s $MODELDIR/model.pytorch ] || wget -q -O $MODELDIR/model.pytorch \
    "https://zenodo.org/record/3733896/files/model.pytorch?download=1"
ls -lh $MODELDIR/model.pytorch build/data/dev-v1.1.json $MODELDIR/vocab.txt
```

**Output**:
```
downloading model.pytorch (~1.3 GB) ...
-rw-r--r-- 1 root root 1.3G Jul 18 11:12 build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch
-rw-r--r-- 1 root root 227K Jul 18 11:08 build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt
-rw-r--r-- 1 root root 4.7M Jul 18 11:08 build/data/dev-v1.1.json
```

### Execution (2026-07-18 11:12:05)
```python
%%writefile tokenization.py
# Minimal self-contained tokenization for the MLPerf BERT/SQuAD reference harness.
# Provides: convert_to_unicode, printable_text, whitespace_tokenize, BasicTokenizer.
# Derived from google-research/bert tokenization.py (Apache-2.0); TensorFlow and the
# vocab/Wordpiece/FullTokenizer pieces removed (the harness uses transformers.BertTokenizer
# for wordpiece and imports only these symbols from this module).
import unicodedata


def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    return text.split()


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        return whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = [c for c in text if unicodedata.category(c) != "Mn"]
        return "".join(output)

    def _run_split_on_punc(self, text):
        chars = list(text)
        i, start_new_word, output = 0, True, []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        return (
            (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)
        )

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            output.append(" " if _is_whitespace(char) else char)
        return "".join(output)


def _is_whitespace(char):
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def _is_control(char):
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char) in ("Cc", "Cf")


def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")
```

**Output**:
```
Writing tokenization.py
```

### Execution (2026-07-18 11:12:05)
```python
%%writefile sitecustomize.py
# OPTIONAL — only needed on NVIDIA Blackwell GPUs (RTX 50-series, sm_120), where
# transformers' fused SDPA attention kernel raises CUDA "illegal instruction".
# Harmless on Colab's T4 / L4 / A100. Auto-imported by the run.py subprocess
# (cwd is on sys.path) so it forces the pure-math SDP backend everywhere.
try:
    import torch
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[sitecustomize] math SDP forced (flash/mem-efficient disabled)", flush=True)
except Exception as e:
    print("[sitecustomize] SDP setup skipped:", e, flush=True)
```

**Output**:
```
Writing sitecustomize.py
```

### Execution (2026-07-18 11:21:59)
```python
import subprocess
cmd = r'''
cd /content/inference/language/bert
python run.py --backend=pytorch --scenario=Offline --accuracy --max_examples=1000 || true
echo "----- f1 / exact_match -----"
python accuracy-squad.py \
  --vocab_file build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt \
  --val_data build/data/dev-v1.1.json \
  --log_file build/logs/mlperf_log_accuracy.json \
  --out_file build/logs/predictions.json \
  --max_examples 1000
'''
r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(r.stdout[-1800:])
print("=== STDERR tail ===")
print(r.stderr[-600:])

```

**Output**:
```

No warnings encountered during test.

No errors encountered during test.
Reading examples...
Loading cached features from 'eval_features.pickle'...
Loading LoadGen logs...
Post-processing predictions...
Writing predictions to: build/result/predictions.json
Loading BERT configs...
Loading PyTorch model...
Constructing SUT...
Finished constructing SUT.
Constructing QSL...
Loading cached features from 'eval_features.pickle'...
Finished constructing QSL.
Running LoadGen test...
----- f1 / exact_match -----
{"exact_match": 86.6, "f1": 90.40153169843566}
Reading examples...
Loading cached features from 'eval_features.pickle'...
Loading LoadGen logs...
Post-processing predictions...
Writing predictions to: build/logs/predictions.json
Evaluating predictions...

=== STDERR tail ===
guage/bert/accuracy-squad.py --max_examples 1000' returned non-zero exit status 1.
/content/inference/language/bert/accuracy-squad.py:27: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  import pkg_resources
2026-07-18 11:21:49.528685: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

```


# Local computer and operational limits

## User-reported hardware

| Item | Specification |
|---|---|
| GPU | NVIDIA RTX 5060 Ti (Blackwell) |
| VRAM | 16 GB |
| System RAM | 32 GB |
| CPU | Intel Core i5-12400F (6 cores/12 threads) |
| Recommended training | Local on Windows + WSL2 Linux, Python isolated env |
| Kaggle run | Inference-only; GPU optional, offline, <=9h |

## WSL setup and GPU sanity check

```powershell
nvidia-smi
wsl --list --verbose
```

In WSL2 install a recent CUDA-enabled **PyTorch that supports compute capability 12.0** from official PyTorch selector, then:

```bash
python scripts/check_gpu.py
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-transformer.txt
```

**Do not hard-code an outdated `pip install torch...` command.** First inspect driver `nvidia-smi`, `torch.__version__`, `torch.version.cuda`, `torch.cuda.get_device_capability()`, and verify a small GPU matmul/backward training. Store the exact working package versions.

## Preflight and resource budgeting (not measured)

- Baseline TF-IDF (CPU): minutes; low/no VRAM.
- Base-size encoder fine-tuning (one model, 5 folds): roughly **1–4 hours** on a suitable local setup, but not a promise; measure one fold first.
- Base encoder full precision FP16 weights: ~0.3–0.4 GB per checkpoint; 5-fold model ensemble may occupy multiple GB.
- Reserve **30–50 GB free working space** for dataset/cache/optimizer/checkpoints during experiments; adjust after actual measurement.
- Single 16 GB GPU requires sequential training. Mixed precision and dynamic padding; checkpointing if needed.
- RAM 32 GB allows sparse TF-IDF and preprocessing; watch pandas memory copies and DataLoader workers.

## Benchmark record fields

- GPU, VRAM, OS, Python, `torch`, CUDA build, driver, BF16 availability.
- Model/checkpoint revision and license, batch size, context length, gradient accumulation, gradient checkpointing.
- Throughput samples/sec, epochs/steps, wall time, max VRAM GB, CPU RSS GB, checkpoint size.
- OOF BA, per-class recall, and fold variability; inference ms/passage on Kaggle-like CPU/GPU.

## Kaggle compatibility

- Inference may have GPU but might differ from RTX 5060 Ti. Avoid architecture-specific compiled artifacts and require graceful device selection.
- Bundle model weights/tokenizer in private Kaggle input; `from_pretrained(..., local_files_only=True)` for transformer.
- Train environment and inference environment need not match exactly, but must use compatible serialization/tokenizers/framework versions.

## 2026-10-18 — Measured local inventory

See `reports/p0_20261018_environment.json`: Windows 11 Pro, i5-12400F (6 cores,
12 logical processors), 31.84 GiB RAM, approximately 174.09 GiB free disk at
inspection, RTX 5060 Ti (16311 MiB reported by nvidia-smi), driver 610.74.
Ubuntu and docker-desktop are installed as WSL2 and were stopped; neither was
started or modified. The CPU baseline uses a separate repository `.venv`.

An existing Conda `gpu_test` environment has Python 3.11.14 and
PyTorch 2.9.1+cu128. Actual import reported CUDA 12.8, capability (12,0), compiled
`sm_120`, and BF16 availability. This inventory is metadata, not a backward or
encoder-speed benchmark. No CUDA packages were installed during this audit.
The original installation command/hash is unknown; do not invent it.

Later on 2026-10-18, after P0, `scripts/check_gpu.py --steps 100 --size 1024
--warmup 10 --seed 42` succeeded in that existing environment. Measured loop:
0.0734684 s; entire check: 4.6000688 s; peak allocated 92277760 bytes, reserved
113246208 bytes; all losses/gradients finite. See
`reports/p1_20261018_gpu_benchmark.json` for exact executable/command and source hash.
Driver was then 610.88, a separate observation from the initial 610.74 inventory;
this work did not install/change a GPU driver. The check uses a synthetic matmul,
so encoder throughput, practical batch size and license review remain unmeasured.

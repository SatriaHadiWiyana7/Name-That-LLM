#!/usr/bin/env python3
"""Offline CUDA inspection and a small deterministic matmul/backward benchmark.

The default remains a one-step GPU check. For the P1 readiness gate, use
--steps 100 --size 1024 --warmup 10 --seed 42 --output <new-report.json>.
No project imports, downloads, pretrained models, or package installs are used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def cpu_memory():
    """Current RSS and OS process peak, when available; never substitute estimates."""
    result = {"rss_bytes": None, "process_peak_rss_bytes": None, "source": None}
    try:
        import psutil

        info = psutil.Process().memory_info()
        result.update(rss_bytes=info.rss, source="psutil.Process.memory_info")
        if hasattr(info, "peak_wset"):
            result["process_peak_rss_bytes"] = info.peak_wset
    except (ImportError, OSError):
        pass
    if result["process_peak_rss_bytes"] is None:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["process_peak_rss_bytes"] = int(
                peak if sys.platform == "darwin" else peak * 1024
            )
            result["peak_source"] = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
        except (ImportError, OSError):
            pass
    return result


def installed_versions():
    versions = {}
    for package in ("torch", "psutil", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def driver_inventory():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = {"command": command}
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=5)
        result.update(
            returncode=process.returncode,
            stdout=process.stdout.strip(),
            stderr=process.stderr.strip(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_cuda(args, report):
    # Set before importing torch or initializing CUDA. This affects only this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    try:
        import torch
    except ImportError as exc:
        report.update(status="torch_unavailable", error=f"{type(exc).__name__}: {exc}")
        return 1

    report["torch"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not report["torch"]["cuda_available"]:
        report.update(status="cuda_unavailable", error="PyTorch cannot access a CUDA device.")
        return 1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report["torch"].update(
        device_name=properties.name,
        device_index=0,
        capability=list(torch.cuda.get_device_capability(device)),
        compiled_arch_list=torch.cuda.get_arch_list(),
        total_vram_bytes=properties.total_memory,
        bf16_supported=torch.cuda.is_bf16_supported(),
        deterministic_algorithms=True,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=False,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )

    x = torch.randn((args.size, args.size), device=device, dtype=torch.float32)
    x.requires_grad_(True)
    scale = math.sqrt(args.size)

    def step():
        x.grad = None
        loss = ((x @ x) / scale).square().mean()
        loss.backward()
        finite = torch.isfinite(loss) & torch.isfinite(x.grad).all()
        return loss, finite

    warmup_started = time.perf_counter()
    warmup_finite = torch.ones((), dtype=torch.bool, device=device)
    for _ in range(args.warmup):
        _, finite = step()
        warmup_finite.logical_and_(finite)
    torch.cuda.synchronize(device)
    report["warmup_elapsed_seconds"] = time.perf_counter() - warmup_started
    if not bool(warmup_finite.item()):
        raise RuntimeError("Non-finite loss or gradient during warmup.")

    x.grad = None
    all_finite = torch.ones((), dtype=torch.bool, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    report["cpu_memory_before_measured_steps"] = cpu_memory()
    started = time.perf_counter()
    completed = 0
    try:
        for _ in range(args.steps):
            loss, finite = step()
            all_finite.logical_and_(finite)
            completed += 1
        torch.cuda.synchronize(device)
    finally:
        elapsed = time.perf_counter() - started
        report["benchmark"] = {
            "completed_steps": completed,
            "elapsed_seconds": elapsed,
            "steps_per_second": completed / elapsed if elapsed > 0 else None,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "timing_includes": "float32 matmul, loss, backward, finite checks, and final CUDA synchronization",
            "timing_excludes": "imports, CUDA setup, tensor initialization, warmup, and JSON writing",
            "workload": "loss = (((x @ x) / sqrt(size)) ** 2).mean(); backward; reset gradient each step",
            "optimizer_steps": 0,
            "scope": "synthetic kernel readiness; not encoder throughput or attribution accuracy",
        }
        report["cpu_memory_after_measured_steps"] = cpu_memory()
    last_loss = float(loss.item())
    last_gradient_abs_max = float(x.grad.abs().max().item())
    report["benchmark"].update(
        all_losses_and_gradients_finite=bool(all_finite.item()),
        last_loss=last_loss if math.isfinite(last_loss) else None,
        last_gradient_abs_max=last_gradient_abs_max if math.isfinite(last_gradient_abs_max) else None,
    )
    if not report["benchmark"]["all_losses_and_gradients_finite"]:
        raise RuntimeError("Non-finite loss or gradient during measured steps.")
    report.update(status="passed", actual_forward_backward_verified=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=1)
    parser.add_argument("--size", type=positive_int, default=128)
    parser.add_argument("--warmup", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--output", type=Path, help="New JSON report path; existing reports are never overwritten.")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists; choose a new versioned path: {args.output}")

    started = time.perf_counter()
    source = Path(__file__)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "actual_forward_backward_verified": False,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "argv": [sys.executable, *sys.argv],
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count()},
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependencies": installed_versions(),
        "configuration": {"steps": args.steps, "size": args.size, "warmup": args.warmup, "seed": args.seed, "dtype": "float32"},
        "cpu_memory_at_start": cpu_memory(),
        "driver_inventory": driver_inventory(),
        "memory_measurement_note": "GPU peaks cover measured steps after warmup reset; reserved cache may include warmup. CPU peak is OS process lifetime, not a GPU-only interval.",
    }
    try:
        returncode = check_cuda(args, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        returncode = 1
    report["total_elapsed_seconds"] = time.perf_counter() - started
    report["cpu_memory_at_end"] = cpu_memory()
    report["exit_code"] = returncode
    serialized = json.dumps(report, indent=2, allow_nan=False)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

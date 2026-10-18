"""Resource measurements and reproducibility checks for local CPU experiments."""

import importlib.metadata
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_info

from .labels import LABEL_SET


class ResourceMonitor:
    """Sample this process's RSS; OS peak counters cover its whole lifetime."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_rss_bytes = 0
        self.os_peak_rss_bytes = None
        self.sample_count = 0
        self._stop = threading.Event()

    def _sample(self):
        memory = self.process.memory_info()
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(memory.rss))
        self.sample_count += 1
        peak = getattr(memory, "peak_wset", None)
        if peak is not None:
            self.os_peak_rss_bytes = max(self.os_peak_rss_bytes or 0, int(peak))

    def _sample_loop(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        self.wall_started = time.perf_counter()
        self.cpu_started = time.process_time()
        self._sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self._sample()
        self.wall_seconds = time.perf_counter() - self.wall_started
        self.cpu_seconds = time.process_time() - self.cpu_started
        # Unix ru_maxrss is a lifetime high-water mark, not a per-phase value.
        if self.os_peak_rss_bytes is None:
            try:
                import resource

                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                self.os_peak_rss_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
            except (ImportError, AttributeError):
                pass

    def report(self) -> dict:
        return {
            "wall_seconds": round(self.wall_seconds, 6),
            "cpu_seconds": round(self.cpu_seconds, 6),
            "peak_rss_sampled_bytes": self.peak_rss_bytes,
            "rss_sampling_interval_seconds": self.interval_seconds,
            "rss_sample_count": self.sample_count,
            "os_peak_rss_bytes": self.os_peak_rss_bytes,
            "os_peak_rss_scope": "process lifetime, not isolated to this phase",
            "gpu_used": False,
            "peak_vram_bytes": None,
            "gpu_note": "CPU scikit-learn pipeline; GPU process memory was not measured",
        }


def environment_snapshot() -> dict:
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": {
            "processor": platform.processor(),
            "identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "system_ram_bytes": int(psutil.virtual_memory().total),
        "dependencies": {
            package.metadata["Name"]: package.version
            for package in sorted(importlib.metadata.distributions(), key=lambda item: item.metadata["Name"].lower())
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
        "threadpools": threadpool_info(),
        "argv": [sys.executable, *sys.argv],
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
    }


def validate_fold_mapping(train: pd.DataFrame, mapping: pd.DataFrame, n_splits: int) -> np.ndarray:
    """Align a complete saved mapping to train rows and reject prompt leakage."""
    required = ["id", "prompt_id", "fold"]
    if not set(required).issubset(mapping.columns):
        raise ValueError("Fold mapping requires id,prompt_id,fold columns")
    if mapping[required].isna().any().any():
        raise ValueError("Fold mapping contains null values")
    if mapping.id.duplicated().any() or train.id.duplicated().any():
        raise ValueError("Fold mapping and train IDs must be unique")
    if len(mapping) != len(train) or set(mapping.id) != set(train.id):
        raise ValueError("Fold mapping IDs must cover exactly the training IDs")
    aligned = mapping.set_index("id").loc[train.id]
    if not np.array_equal(aligned.prompt_id.to_numpy(), train.prompt_id.to_numpy()):
        raise ValueError("Fold mapping prompt IDs differ from training data")
    numeric = pd.to_numeric(aligned.fold, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Fold numbers must be finite integers")
    if set(numeric) != set(range(n_splits)):
        raise ValueError("Fold numbers must cover exactly 0 through n_splits-1")
    assigned = numeric.astype(int)
    if pd.DataFrame({"prompt_id": train.prompt_id.to_numpy(), "fold": assigned}).groupby("prompt_id").fold.nunique().max() != 1:
        raise ValueError("PROMPT LEAKAGE: a prompt occurs in multiple validation folds")
    for fold in range(n_splits):
        validation = assigned == fold
        if set(train.loc[validation, "model"]) != LABEL_SET:
            raise ValueError(f"Validation fold {fold} is missing legal labels")
        if set(train.loc[~validation, "model"]) != LABEL_SET:
            raise ValueError(f"Training fold {fold} is missing legal labels")
        if set(train.loc[validation, "prompt_id"]) & set(train.loc[~validation, "prompt_id"]):
            raise ValueError(f"PROMPT LEAKAGE in fold {fold}")
    return assigned


def reference_comparison(current: dict, reference: dict) -> dict:
    """Only comparable provenance permits a delta intended as an experiment gain."""
    required = ("dataset_sha256", "fold_assignments_sha256", "random_seed")
    comparable = all(key in reference and current[key] == reference[key] for key in required)
    difference = float(current["balanced_accuracy"] - reference["balanced_accuracy"])
    result = {
        "reference_balanced_accuracy": float(reference["balanced_accuracy"]),
        "numerical_difference": difference,
        "identical_dataset_folds_seed_verified": comparable,
        "delta_balanced_accuracy": difference if comparable else None,
        "interpretation": (
            "Measured difference on identical dataset hash, saved fold mapping hash and seed"
            if comparable else
            "Historical numerical difference only; identical dataset/folds/seed not verified; not a measured improvement"
        ),
    }
    if comparable:
        previous_folds = {item["fold"]: item for item in reference.get("folds", [])}
        result["per_fold_delta_balanced_accuracy"] = {
            str(item["fold"]): item["balanced_accuracy"] - previous_folds[item["fold"]]["balanced_accuracy"]
            for item in current["folds"] if item["fold"] in previous_folds
        }
        result["class_recall_delta"] = {
            label: recall - reference["class_recall"][label]
            for label, recall in current["class_recall"].items()
        }
    return result

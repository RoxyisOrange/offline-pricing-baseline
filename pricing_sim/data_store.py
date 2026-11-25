"""Write-once storage for the 100 paired simulation datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .simulation import OBSERVED_COLUMNS, data_statistics, generate_dataset

NAMES = ("observed_train", "observed_test", "truth_train", "truth_test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def freeze_data(config_path: Path, output: Path) -> dict:
    """Create a new data version; never overwrite an existing one."""
    if output.exists():
        raise FileExistsError(output)
    cfg = load_config(config_path)
    (output / "runs").mkdir(parents=True)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    entries, statistics = [], []
    root = np.random.SeedSequence(cfg["run"]["seed"])
    for run, pair in enumerate(root.spawn(cfg["simulation"]["n_repeats"])):
        train_seed, test_seed, fit_seed = [int(s.generate_state(1)[0]) for s in pair.spawn(3)]
        train, train_truth = generate_dataset(cfg, train_seed, cfg["simulation"]["n_train"])
        test, test_truth = generate_dataset(cfg, test_seed, cfg["simulation"]["n_test"])
        run_dir = output / "runs" / f"run_{run:03d}"
        run_dir.mkdir()
        files = {}
        for name, frame in zip(NAMES, (train, test, train_truth, test_truth)):
            path = run_dir / f"{name}.csv.gz"
            frame.to_csv(path, index=False, float_format="%.17g",
                         compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
            files[name] = {"path": str(path.relative_to(output)), "sha256": sha256(path)}
        for split, truth in (("train", train_truth), ("test", test_truth)):
            stats = data_statistics(truth)
            statistics.append({"run": run, "split": split, "n": len(truth),
                               "inventory_mean": stats["distributions"]["inventory"]["mean"],
                               "demand_mean": stats["distributions"]["true_demand"]["mean"],
                               "censoring_indicator_rate": stats["censoring_indicator_rate"],
                               "tie_rate": stats["tie_rate"],
                               "unmet_demand_rate": stats["unmet_demand_rate"]})
        entries.append({"run": run, "train_seed": train_seed, "test_seed": test_seed,
                        "fit_seed": fit_seed, "files": files})
    pd.DataFrame(statistics).to_csv(output / "statistics.csv", index=False)
    manifest = {"schema_version": 1, "n_repeats": len(entries),
                "seed_plan": "SeedSequence(root).spawn(repeats); pair.spawn(3)",
                "config_sha256": sha256(output / "config.yaml"),
                "statistics_sha256": sha256(output / "statistics.csv"), "runs": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(data_dir: Path, verify_hashes: bool = True) -> dict:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if sha256(data_dir / "config.yaml") != manifest["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    if sha256(data_dir / "statistics.csv") != manifest["statistics_sha256"]:
        raise ValueError("Frozen statistics changed")
    if len(manifest["runs"]) != manifest["n_repeats"]:
        raise ValueError("Incomplete data manifest")
    if verify_hashes:
        for entry in manifest["runs"]:
            for name in NAMES:
                info = entry["files"][name]
                if sha256(data_dir / info["path"]) != info["sha256"]:
                    raise ValueError(f"Data changed: run {entry['run']} {name}")
    return manifest


def load_run(data_dir: Path, run: int, include_truth: bool = False):
    """Return observed train/test only unless truth is explicitly requested."""
    entry = load_manifest(data_dir, verify_hashes=False)["runs"][run]
    names = NAMES if include_truth else NAMES[:2]
    frames = tuple(pd.read_csv(data_dir / entry["files"][name]["path"],
                               compression="gzip", float_precision="round_trip") for name in names)
    for frame in frames[:2]:
        if list(frame.columns) != OBSERVED_COLUMNS:
            raise ValueError("Observed schema changed")
    return frames

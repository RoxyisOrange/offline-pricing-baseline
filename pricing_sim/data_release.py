"""CSV layout and integrity metadata for the shared data release and frozen E0 copy."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import tempfile

import numpy as np
import pandas as pd
import scipy

from .data_store import load_config, load_manifest, load_run, sha256
from .experiment_data import (
    EXPERIMENTS, GROUP_THRESHOLD, GROUPS, N_TEST, N_TRAIN, SCENARIOS, TEST_COLUMNS,
    VERSION, _DESIGN, _check_round, generate_diagnostic_test, generate_test, generate_train,
)

LAYOUT_VERSION = 2
GENERATION_CONFIG = Path("metadata/generation_config.json")
QUALITY_REPORT = Path("metadata/data_quality_report.csv")
CHECKSUMS = Path("metadata/checksums.sha256")
RELEASE_ENTRIES = ("train", "test", "ground_truth", "README.md", "metadata")
E0 = "E0"


def scenario_folder(scenario_id: str) -> Path:
    if scenario_id == E0:
        return Path("poisson", E0)
    scenario = SCENARIOS[scenario_id]
    return Path(scenario.market, f"{scenario_id}_L{scenario.low_target_pct}_H{scenario.high_target_pct}")


def _destination_available(output: Path) -> None:
    for name in (*RELEASE_ENTRIES, "manifest.json"):
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Data release already has {path}")


def publish_release(staged: Path, output: Path) -> None:
    """Install the release alongside existing archives, with the manifest arriving last."""
    staged = Path(staged)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".release-installing"
    lock.mkdir()
    moved = []
    try:
        _destination_available(output)
        for name in RELEASE_ENTRIES:
            (staged / name).rename(output / name)
            moved.append(name)
    except BaseException:
        for name in reversed(moved):
            (output / name).rename(staged / name)
        raise
    finally:
        lock.rmdir()


@contextmanager
def _staged_release(output: Path):
    _destination_available(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".csv-release-", dir=output.parent) as temp:
        stage = Path(temp)
        writer = _Writer(stage)
        yield writer
        publish_release(stage, output)


class _Writer:
    def __init__(self, root: Path):
        self.root = root
        self.files = {}

    def write(self, relative: Path, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False, float_format="%.17g")
        else:
            path.write_text(content, encoding="utf-8")
        self.files[relative.as_posix()] = sha256(path)

    def complete(self, config, scenarios, mapping, calibration, quality, readme):
        self.write(Path("metadata/scenario_manifest.csv"), pd.DataFrame(scenarios))
        self.write(Path("metadata/experiment_mapping.json"), json.dumps(mapping, indent=2) + "\n")
        self.write(Path("metadata/calibration_parameters.csv"), pd.DataFrame(calibration))
        self.write(QUALITY_REPORT, pd.DataFrame(quality))
        self.write(GENERATION_CONFIG, json.dumps(config, indent=2) + "\n")
        self.write(Path("README.md"), readme)
        (self.root / CHECKSUMS).write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(self.files.items())), encoding="utf-8"
        )


def read_release(root: Path) -> tuple[dict, str]:
    config = (root / GENERATION_CONFIG).read_bytes()
    listed = (root / CHECKSUMS).read_bytes()
    files = {}
    for line in listed.decode("utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = PurePosixPath(name)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or path.as_posix() != name
                or path.is_absolute() or ".." in path.parts or name in files):
            raise ValueError(f"Invalid release checksum entry: {name}")
        files[name] = digest
    if files.get(GENERATION_CONFIG.as_posix()) != hashlib.sha256(config).hexdigest():
        raise ValueError("Frozen generation configuration changed")
    manifest = json.loads(config)
    if manifest["layout_version"] != LAYOUT_VERSION:
        raise ValueError("Unsupported CSV release layout")
    manifest["files"] = files
    return manifest, hashlib.sha256(config + b"\0" + listed).hexdigest()


def _versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__}


def _scenario_record(sid, rounds, n_train, n_test):
    spec = SCENARIOS.get(sid)
    directory = scenario_folder(sid)
    return {
        "scenario_id": sid, "market": directory.parts[0],
        "low_target_pct": spec.low_target_pct if spec else None,
        "high_target_pct": spec.high_target_pct if spec else None,
        "directory": directory.as_posix(),
        "train_path": (Path("train") / directory).as_posix(),
        "truth_path": (Path("ground_truth") / directory).as_posix(),
        "rounds": rounds, "n_train": n_train, "n_test": n_test,
    }


def _portable_truth(truth):
    return pd.DataFrame({
        "sample_id": truth.sample_id,
        "demand": truth.true_demand,
        "lambda": truth.demand_rate,
        "mean_demand": 1 + truth.demand_rate,
    })


def _quality(sid, rep, observed, truth, test):
    if not truth.sample_id.equals(observed.sample_id):
        raise ValueError(f"Training observations and truth do not align: {sid}, round {rep}")
    if (
        not np.array_equal(observed.sales, np.minimum(truth.demand, observed.inventory))
        or not np.array_equal(observed.delta, (truth.demand < observed.inventory).astype(int))
    ):
        raise ValueError(f"Invalid sales or censoring: {sid}, round {rep}")
    missing = sum(int(frame.isna().sum().sum()) for frame in (observed, truth, test))
    if missing:
        raise ValueError(f"Missing values: {sid}, round {rep}")
    row = {
        "scenario_id": sid, "market": scenario_folder(sid).parts[0], "round": rep,
        "n_train": len(observed), "n_test": len(test), "missing_values": missing,
        "actual_rate": (observed.delta == 0).mean(),
        "zero_inventory_rate": (observed.inventory == 0).mean(),
        "inventory_at_most_one_rate": (observed.inventory <= 1).mean(),
        "train_inventory_min": observed.inventory.min(), "train_inventory_max": observed.inventory.max(),
        "test_inventory_min": test.inventory.min(), "test_inventory_max": test.inventory.max(),
        "test_above_train_max_rate": (test.inventory > observed.inventory.max()).mean(),
    }
    for label in GROUPS:
        group = observed[observed.group == label]
        test_group = test[test.group == label]
        if group.empty or test_group.empty:
            raise ValueError(f"Missing {label} group: {sid}, round {rep}")
        low_high = label.lower()
        row.update({
            f"n_{low_high}": len(group), f"n_test_{low_high}": len(test_group),
            f"actual_{low_high}_rate": (group.delta == 0).mean(),
            f"train_{low_high}_inventory_max": group.inventory.max(),
            f"test_{low_high}_above_train_max_rate": (test_group.inventory > group.inventory.max()).mean(),
        })
    return row


def _write_train(writer, sid, rep, observed, truth, test):
    folder = scenario_folder(sid)
    truth = _portable_truth(truth)
    quality = _quality(sid, rep, observed, truth, test)
    writer.write(Path("train") / folder / f"train_{rep:03d}.csv", observed)
    writer.write(Path("ground_truth") / folder / f"truth_{rep:03d}.csv", truth)
    return quality


def _write_test(writer, rep, test, diagnostics):
    writer.write(Path("test/shared", f"test_{rep:03d}.csv"), test)
    for market, observed in diagnostics.items():
        if not test.equals(observed[list(TEST_COLUMNS)]):
            raise ValueError(f"Diagnostic and shared test rows differ for {market}, round {rep}")
        writer.write(Path("test/diagnostics", market, f"diagnostic_{rep:03d}.csv"), observed)


def export_experiments(output: Path, rounds: int, scenarios, *, code_revision=None) -> dict:
    rounds = _check_round(rounds)
    scenarios = tuple(dict.fromkeys(scenarios))
    if not scenarios or set(scenarios).difference(SCENARIOS):
        raise ValueError("Choose at least one registered scenario")
    records = [_scenario_record(sid, rounds, N_TRAIN, N_TEST) for sid in scenarios]
    markets = sorted({SCENARIOS[sid].market for sid in scenarios})
    mapping = {exp: [sid for sid in members if sid in scenarios] for exp, members in EXPERIMENTS.items()}
    quality, calibration = [], {}
    config = {
        "dataset": "E1-E4", "layout_version": LAYOUT_VERSION, "version": VERSION,
        "rounds": rounds, **_DESIGN, "code_revision": code_revision,
        "generator_sha256": sha256(Path(__file__).with_name("experiment_data.py")),
        "exporter_sha256": sha256(Path(__file__)), "versions": _versions(),
        "demand_rate": {"intercept": 5, "x1_squared": 1, "x2_squared": 1, "price_coefficient": -5},
        "demand_shift": 1, "price_bounds": [0, 1], "inventory_rule": "nonnegative normal, then floor",
        "censor_indicator": "delta = 1(sales < inventory)",
        "scenarios": {row["scenario_id"]: row for row in records}, "experiments": mapping,
    }
    with _staged_release(Path(output)) as writer:
        for rep in range(1, rounds + 1):
            test = generate_test(rep)
            _write_test(writer, rep, test, {market: generate_diagnostic_test(rep, market) for market in markets})
            for sid in scenarios:
                sample = generate_train(sid, rep)
                row = _write_train(writer, sid, rep, sample.observed, sample.truth, test)
                for group in GROUPS:
                    label = group.lower()
                    if row[f"n_{label}"] != N_TRAIN // 2 or row[f"n_test_{label}"] != N_TEST // 2:
                        raise ValueError(f"Invalid stratum size: {sid}, round {rep}")
                    target = sample.statistics[f"r_{label}_target"]
                    mean = sample.statistics[f"inventory_mean_parameter_{label}"]
                    key = (row["market"], group, target)
                    item = calibration.setdefault(key, {
                        "market": row["market"], "group": group, "target_rate": target,
                        "inventory_mean_parameter": mean, "source": "Sobol calibration",
                    })
                    if item["inventory_mean_parameter"] != mean:
                        raise ValueError(f"Inventory calibration changed for {key}")
                quality.append(row)
        writer.complete(config, records, mapping, calibration.values(), quality, _shared_readme(rounds, scenarios))
    return config


def _shared_readme(rounds, scenarios):
    return f"""# E1–E4 公用数据

本目录有 {len(scenarios)} 个训练场景和 {rounds} 轮，每份训练集、共享测试集均有 2000 行，Low 和 High 各 1000 行。

- `train/` 按市场及场景保存观测训练数据。同一场景和轮次供所有方法复用。
- `test/shared/` 每轮一份，共用商品特征和库存。
- `test/diagnostics/` 保存两个市场各自的评价附表。
- `ground_truth/` 保存仅用于评价和核验的训练真值。按场景、轮次和 `sample_id` 对齐。
- `metadata/` 保存参数、校准、逐轮数据质量和校验和。库存覆盖率按当轮训练样本的最大库存计算。

组员可调用仓库 `data_loader.py` 的 `load_train`、`load_test` 和 `get_scenarios`。
普通训练读取 `train/` 的观测列。共享测试、诊断和真值属于评价环节。
在本目录运行 `shasum -a 256 -c metadata/checksums.sha256` 核验本次发布文件。
E0 CSV 副本位于单独的 `e0_frozen/`。
"""


def _annotate_e0(observed):
    group = np.where(observed.x1**2 + observed.x2**2 <= GROUP_THRESHOLD, *GROUPS)
    return observed.assign(group=group)


def export_e0(source: Path, output: Path, *, source_revision=None) -> dict:
    """Convert verified original E0 rows into a separate read-only CSV copy."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"E0 archive already exists: {output}")
    legacy = load_manifest(source)
    legacy_config = load_config(source / "config.yaml")
    simulation = legacy_config["simulation"]
    rounds = legacy["n_repeats"]
    record = _scenario_record(E0, rounds, simulation["n_train"], simulation["n_test"])
    mapping = {E0: [E0]}
    config = {
        "dataset": E0, "layout_version": LAYOUT_VERSION, "version": "baseline-v1",
        "source_revision": source_revision, "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_seed_plan": legacy["seed_plan"],
        "source_runs": [
            {key: run[key] for key in ("run", "train_seed", "test_seed", "fit_seed")} for run in legacy["runs"]
        ],
        "legacy_config": legacy_config, "rounds": rounds, "n_train": record["n_train"], "n_test": record["n_test"],
        "group_threshold": GROUP_THRESHOLD, "versions": _versions(), "scenarios": {E0: record}, "experiments": mapping,
    }
    calibration = [{
        "market": "poisson", "group": label, "target_rate": None,
        "inventory_mean_parameter": simulation["inventory_mean"], "source": "fixed E0 mechanism",
    } for label in GROUPS]
    readme = """# E0 baseline-v1 冻结 CSV 副本

数据来自已校验的 `data/v1` 历史 E0。复制时保留原始行顺序、样本 ID 和数值。
原来的 `run_000` 对应本目录第 1 轮，依次到第 100 轮。
Low 和 High 由原特征打标签，人数沿用历史 E0 的自然分组。
测试主表和 Poisson 诊断附表使用原来同一批 E0 测试记录。
本副本可由公共读取器读取。存档训练和结果仍由 `baseline-v1` 标识。

在本目录运行 `shasum -a 256 -c metadata/checksums.sha256` 核验副本。
"""
    with _staged_release(output) as writer:
        quality = []
        for rep in range(1, rounds + 1):
            train, observed_test, true_train, _ = load_run(source, rep - 1, include_truth=True)
            train = _annotate_e0(train)
            diagnostic = _annotate_e0(observed_test)
            test = diagnostic[list(TEST_COLUMNS)]
            _write_test(writer, rep, test, {"poisson": diagnostic})
            quality.append(_write_train(writer, E0, rep, train, true_train, test))
        writer.complete(config, [record], mapping, calibration, quality, readme)
    for path in output.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    output.chmod(0o555)
    return config

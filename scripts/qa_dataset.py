"""Scan published CSV files, replay their random streams and audit their provenance."""

from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
from typing import Optional
import platform
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr

from data_loader import open_dataset
from pricing_sim.data_store import load_manifest, load_run
from pricing_sim.experiment_data import export_data
from qa_statistics import Coverage, check_calibration
from train import _shared_training_inputs

OBSERVED = ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta", "group"]
TEST = ["sample_id", "x1", "x2", "inventory", "group"]
TRUTH = ["sample_id", "demand", "lambda", "mean_demand"]
INTEGER = {"sample_id", "inventory", "sales", "delta", "demand"}
PAIR_COLUMNS = ["sample_id", "x1", "x2", "price", "group"]
GROUPS = ("Low", "High")
MARKETS = ("poisson", "negative_binomial")
ID = "metadata/generation_config.json"
HASH_LIST = "metadata/checksums.sha256"
META = {
    ID, "metadata/experiment_mapping.json", "metadata/scenario_manifest.csv",
    "metadata/calibration_parameters.csv", "metadata/data_quality_report.csv", "README.md",
}
TARGETS = (
    (10, 10), (30, 30), (50, 50), (70, 70), (90, 90), (40, 60), (30, 70),
    (20, 80), (10, 90), (20, 40), (10, 50), (60, 80), (50, 90),
    (10, 10), (50, 50), (90, 90), (30, 70), (10, 90),
)
PLAN = {f"S{i:02d}": (MARKETS[i > 13], *pair) for i, pair in enumerate(TARGETS, 1)}
MAPPING = {
    "E1": ["S01", "S02", "S03", "S04", "S05"],
    "E2": ["S03", "S06", "S07", "S08", "S09"],
    "E3": ["S02", "S10", "S11", "S03", "S06", "S07", "S04", "S12", "S13"],
    "E4.1": ["S01", "S03", "S05", "S07", "S09"],
    "E4.2": ["S14", "S15", "S16", "S17", "S18"],
}
SEED = 20261125
THRESHOLD = 2 / np.pi
IDENTIFIERS = np.arange(1, 2001)
FIXED = {
    "dataset": "E1-E4", "version": "2.0.0", "layout_version": 2, "rounds": 100,
    "master_seed": SEED, "pilot_seed": 20261126, "pilot_log2_n": 17,
    "n_train": 2000, "n_test": 2000, "n_per_train_group": 1000, "n_per_test_group": 1000,
    "group_threshold": THRESHOLD, "price_mean": .5, "price_sd": .5, "price_bounds": [0, 1],
    "inventory_mean_test": 6.0, "inventory_variance": 2.0, "nb_size": 2.0,
    "stockout_cost": .1, "oracle_step": .001, "demand_shift": 1,
    "demand_rate": {"intercept": 5, "x1_squared": 1, "x2_squared": 1, "price_coefficient": -5},
    "inventory_rule": "nonnegative normal, then floor", "censor_indicator": "delta = 1(sales < inventory)",
}
ISSUE_COLUMNS = ["check", "severity", "file", "scenario_id", "round_id", "line_number",
                 "sample_id", "field", "value", "reason"]


def file_hash(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class Ref:
    file: str
    sid: str = ""
    rep: Optional[int] = None


class Audit:
    def __init__(self, root, logger):
        self.root = root
        self.logger = logger
        self.issues = []
        self.checksums = []
        self.file_counts = Counter()
        self.row_counts = Counter()

    def issue(self, check, ref, reason, *, field="", value="", sample_id=None,
              line_number=None, severity="FAIL"):
        self.issues.append(dict(zip(ISSUE_COLUMNS, (
            check, severity, ref.file, ref.sid, ref.rep, line_number, sample_id, field, value, reason
        ))))
        if len(self.issues) <= 35:
            self.logger.warning("%s check %s %s rep=%s id=%s %s %s", severity, check,
                                ref.file, ref.rep, sample_id, field, reason)

    def require(self, condition, check, ref, reason, **kwargs):
        if not condition:
            self.issue(check, ref, reason, **kwargs)
        return condition

    def rows(self, valid, check, ref, frame, field, reason):
        ids = frame.sample_id if "sample_id" in frame else frame.index
        for index in np.flatnonzero(~np.asarray(valid, dtype=bool)):
            value = frame[field].iloc[index] if field in frame else ""
            self.issue(check, ref, reason, field=field, value=value, sample_id=ids.iloc[index]
                       if "sample_id" in frame else index, line_number=index + 2)

    def read(self, ref, columns=None, *, category=None, e0=False):
        try:
            frame = pd.read_csv(self.root / ref.file, float_precision="round_trip")
        except (OSError, UnicodeError, ValueError) as error:
            self.issue("01", ref, f"Cannot read CSV: {error}")
            return None
        if category:
            self.file_counts[category] += 1
            self.row_counts[category] += len(frame)
        if columns is not None:
            check = "01" if e0 else "02"
            if not self.require(list(frame) == columns, check, ref, "Unexpected columns or column order",
                                value=",".join(frame.columns)):
                return None
            if not self.require(len(frame) == 2000, check, ref, "Expected 2000 data rows", value=len(frame)):
                return None
            valid_table = True
            for name in columns:
                values = frame[name]
                if name == "group":
                    valid = values.isin(GROUPS)
                    self.rows(valid, check, ref, frame, name, "Invalid group")
                    valid_table &= valid.all()
                    continue
                num = pd.to_numeric(values, errors="coerce")
                valid = np.isfinite(num)
                self.rows(valid, check, ref, frame, name, "Missing, nonnumeric or nonfinite value")
                expected_dtype = values.dtype.kind in ("iu" if name in INTEGER else "f")
                self.require(expected_dtype, check, ref, "Incorrect stored numeric type",
                             field=name, value=str(values.dtype))
                valid_table &= valid.all() and expected_dtype
            if not valid_table:
                return None
            ids = IDENTIFIERS - int(e0)
            self.rows(frame.sample_id.to_numpy() == ids, check, ref, frame, "sample_id",
                      "Missing, duplicated or reordered sample ID")
            for name in ("x1", "x2", "price"):
                if name in frame:
                    self.rows(frame[name].between(0, 1), check, ref, frame, name, "Outside [0,1]")
            for name, floor in (("inventory", 0), ("sales", 0), ("demand", 1), ("lambda", 0), ("mean_demand", 1)):
                if name in frame:
                    self.rows(frame[name] >= floor, check, ref, frame, name, "Below allowed minimum")
            if "sales" in frame:
                self.rows(frame.sales.le(frame.inventory), check, ref, frame, "sales", "Exceeds inventory")
                self.rows(frame.delta.isin([0, 1]), check, ref, frame, "delta", "Not binary")
                self.rows(frame.delta.eq(frame.sales < frame.inventory), check, ref, frame, "delta",
                          "Disagrees with 1(sales<inventory)")
            if "group" in frame:
                self.rows(frame.group.eq(np.where(frame.x1**2 + frame.x2**2 <= THRESHOLD, *GROUPS)),
                          check, ref, frame, "group", "Disagrees with fixed feature threshold")
                self.rows(~frame.duplicated(["x1", "x2"], keep=False), check, ref, frame, "x1",
                          "Repeated joint feature vector within one table")
                if not e0:
                    self.require(frame.group.value_counts().to_dict() == {"Low": 1000, "High": 1000},
                                 check, ref, "Group sizes differ from 1000/1000")
        return frame


def paths(sid, rep):
    market, low, high = PLAN[sid]
    folder = f"{market}/{sid}_L{low}_H{high}"
    return Ref(f"train/{folder}/train_{rep:03d}.csv", sid, rep), Ref(
        f"ground_truth/{folder}/truth_{rep:03d}.csv", sid, rep)


def test_ref(rep):
    return Ref(f"test/shared/test_{rep:03d}.csv", "shared", rep)


def diagnostic_ref(market, rep):
    return Ref(f"test/diagnostics/{market}/diagnostic_{rep:03d}.csv", market, rep)


def official_file_set():
    return {
        ref.file for rep in range(1, 101) for ref in (
            test_ref(rep), *(diagnostic_ref(m, rep) for m in MARKETS),
            *(f for sid in PLAN for f in paths(sid, rep)),
        )
    }


def e0_files():
    return {
        name for rep in range(1, 101) for name in (
            f"train/poisson/E0/train_{rep:03d}.csv", f"ground_truth/poisson/E0/truth_{rep:03d}.csv",
            test_ref(rep).file, diagnostic_ref("poisson", rep).file,
        )
    }


def check_tree(audit, expected, prefix=""):
    folder = audit.root / prefix
    actual = set()
    for name in ("train", "test", "ground_truth"):
        actual.update(p.relative_to(folder).as_posix() for p in (folder / name).rglob("*") if p.is_file())
    for name in sorted(expected - actual):
        audit.issue("01", Ref(prefix + name), "Missing data file")
    for name in sorted(actual - expected):
        audit.issue("01", Ref(prefix + name), "Unexpected data file")
    audited_metadata = {p.relative_to(folder).as_posix() for p in (folder / "metadata").rglob("*") if p.is_file()}
    for name in sorted((META - {"README.md"} | {HASH_LIST}) ^ audited_metadata):
        audit.issue("01", Ref(prefix + name), "Metadata file inventory differs from layout")
    return len(actual)


def verify_checksums(audit, expected, prefix=""):
    recorded = {}
    hash_ref = Ref(prefix + HASH_LIST)
    try:
        catalog = (audit.root / hash_ref.file).read_bytes()
    except OSError as error:
        audit.issue("09", hash_ref, str(error))
        return None
    for line in catalog.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            audit.issue("09", hash_ref, "Malformed hash line", value=line)
            continue
        digest, name = parts
        if name in recorded:
            audit.issue("09", Ref(prefix + name), "Duplicate checksum record")
        recorded[name] = digest
    for name in sorted(set(recorded) - expected):
        audit.issue("09", Ref(prefix + name), "Unexpected checksum record")
    for name in sorted(expected):
        relative = prefix + name
        path = audit.root / relative
        try:
            actual = file_hash(path)
        except OSError:
            actual = ""
        saved = recorded.get(name, "")
        valid = bool(actual and saved == actual)
        audit.checksums.append({"release": "E0" if prefix else "E1-E4", "file": relative,
                                "expected_sha256": saved, "actual_sha256": actual,
                                "status": "PASS" if valid else "FAIL"})
        if not valid:
            audit.issue("09", Ref(relative), "File hash missing or differs from release catalog")
    return sha256((audit.root / (prefix + ID)).read_bytes() + b"\0" + catalog).hexdigest()


def random_stream(rep, split, purpose):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, rep, split, purpose])))


def features_from_seed(rep, split):
    rng = random_stream(rep, split, 10)
    retained = [[], []]
    while min(map(len, retained)) < 1000:
        for row in rng.random((2000, 2)):
            group = int(row[0] * row[0] + row[1] * row[1] > THRESHOLD)
            if len(retained[group]) < 1000:
                retained[group].append(row)
    return rng.permutation(np.concatenate(retained))


def price_cdf(p):
    return (ndtr((p-.5)/.5) - ndtr(-1.)) / (ndtr(1.) - ndtr(-1.))


def inventory_valid(y, mu, quantile):
    sd = np.sqrt(2.)
    base = ndtr(-mu/sd)
    below = (ndtr((y-mu)/sd) - base) / (1-base)
    above = (ndtr((y+1-mu)/sd) - base) / (1-base)
    return (below <= quantile + 3e-14) & (quantile <= above + 3e-14)


def draw_demand(observed, rep, split, market):
    x1, x2, p = (observed[column].to_numpy() for column in ("x1", "x2", "price"))
    rate = 5 + x1*x1 + x2*x2 - 5*p
    if market == "poisson":
        return rate, 1 + random_stream(rep, split, 30).poisson(rate)
    return rate, 1 + random_stream(rep, split, 31).negative_binomial(2, 2/(2+rate))


def verify_context(audit, df, ref, x, rep, split):
    for i, col in enumerate(("x1", "x2")):
        audit.rows(df[col].to_numpy() == x[:, i], "05", ref, df, col,
                   "Does not match independently replayed stratified feature")
    if "price" in df:
        probability = random_stream(rep, split, 20).random(2000)
        good = np.isclose(price_cdf(df.price.to_numpy()), probability, rtol=0, atol=3e-15)
        audit.rows(good, "03", ref, df, "price", "Does not match the truncated normal CDF")


def verify_sales(audit, frame, d, ref, check):
    y = frame.inventory.to_numpy()
    audit.rows(frame.sales.le(d), check, ref, frame, "sales", "Exceeds realized demand")
    audit.rows(frame.sales.eq(np.minimum(d, y)), check, ref, frame, "sales", "Disagrees with min(D,Y)")
    audit.rows(frame.delta.eq(d < y), check, ref, frame, "delta", "Incorrect D<Y indicator or tie")


def compare_columns(audit, right, left, columns, ref, reason, check):
    for col in columns:
        audit.rows(right[col].to_numpy() == left[col].to_numpy(), check, ref, right, col, reason)


def verify_design(audit):
    config = json.loads((audit.root / ID).read_text())
    for key, value in FIXED.items():
        audit.require(config.get(key) == value, "03", Ref(ID), "Parameter differs from the frozen experiment design",
                      field=key, value=config.get(key))
    audit.require(config.get("experiments") == MAPPING, "01", Ref(ID), "Incorrect experiment map")
    mapping_path = "metadata/experiment_mapping.json"
    audit.require(json.loads((audit.root / mapping_path).read_text()) == MAPPING, "01", Ref(mapping_path),
                  "Incorrect scenario assignments or reuse")
    table_path = "metadata/scenario_manifest.csv"
    table = audit.read(Ref(table_path))
    expected_rows = []
    for sid, (market, low, high) in PLAN.items():
        folder = str(Path(paths(sid, 1)[0].file).parent).removeprefix("train/")
        expected_rows.append({"scenario_id": sid, "market": market, "low_target_pct": low,
                              "high_target_pct": high, "directory": folder,
                              "train_path": f"train/{folder}", "truth_path": f"ground_truth/{folder}",
                              "rounds": 100, "n_train": 2000, "n_test": 2000})
    audit.require(table is not None and table.to_dict("records") == expected_rows, "01", Ref(table_path),
                  "Scenario manifest paths, parameters or counts disagree with plan")
    audit.require(list(config["scenarios"].values()) == expected_rows, "01", Ref(ID),
                  "Generation configuration scenario table differs from plan")
    return config


def scan_formal_rows(audit):
    cal_path = "metadata/calibration_parameters.csv"
    cal = audit.read(Ref(cal_path))
    means = cal.set_index(["market", "group", "target_rate"], verify_integrity=True)["inventory_mean_parameter"]
    required = {(market, group, target/100) for market, low, high in PLAN.values()
                for group, target in zip(GROUPS, (low, high))}
    audit.require(set(means.index) == required and np.isfinite(means).all() and len(means) == 24,
                  "03", Ref(cal_path), "Missing, nonfinite or unexpected inventory calibration keys")
    integrals = check_calibration(cal)
    audit.require(bool(integrals) and max(abs(x["deviation_pp"]) for x in integrals) <= 1,
                  "03", Ref(cal_path), "Independent calibration integral exceeds one percentage point")
    quality_ref = Ref("metadata/data_quality_report.csv")
    quality = audit.read(quality_ref).set_index(["scenario_id", "round"], verify_integrity=True)
    audit.require(len(quality) == 1800, "01", quality_ref, "Expected 1800 data quality entries")

    coverage = Coverage()
    per_round = []
    contexts = {"train_shared_once": [], "test_diagnostics_shared_once": []}
    demand_samples = {m: [] for m in MARKETS}
    train_keys, test_keys = set(), set()
    ties = zero_count = 0
    for rep in range(1, 101):
        test = audit.read(test_ref(rep), TEST, category="shared_test")
        if test is not None:
            x_test = features_from_seed(rep, 1)
            verify_context(audit, test, test_ref(rep), x_test, rep, 1)
            valid = inventory_valid(test.inventory.to_numpy(), 6., random_stream(rep, 1, 40).random(2000))
            audit.rows(valid, "03", test_ref(rep), test, "inventory", "Fails fixed test-inventory CDF replay")
            test_keys.add(sha256(test[["x1", "x2"]].to_numpy().tobytes()).hexdigest())
        for market in MARKETS:
            ref = diagnostic_ref(market, rep)
            diagnostic = audit.read(ref, OBSERVED, category="test_diagnostics")
            if diagnostic is None:
                continue
            if test is not None:
                compare_columns(audit, diagnostic, test, TEST, ref, "Test diagnostic and shared context differ", "06")
                verify_context(audit, diagnostic, ref, x_test, rep, 1)
            _, d = draw_demand(diagnostic, rep, 1, market)
            verify_sales(audit, diagnostic, d, ref, "06")
            if market == "poisson":
                contexts["test_diagnostics_shared_once"].append(diagnostic[["x1", "x2", "price", "group"]])

        x_train = features_from_seed(rep, 0)
        quantile = random_stream(rep, 0, 40).random(2000)
        shared, paired_truth = None, {}
        for sid, (market, low, high) in PLAN.items():
            observed_ref, truth_ref = paths(sid, rep)
            train = audit.read(observed_ref, OBSERVED, category="train")
            truth = audit.read(truth_ref, TRUTH, category="training_truth")
            if train is None or truth is None:
                continue
            if shared is None:
                shared = train[PAIR_COLUMNS]
                train_keys.add(sha256(train[["x1", "x2"]].to_numpy().tobytes()).hexdigest())
                contexts["train_shared_once"].append(train[["x1", "x2", "price", "group"]])
                verify_context(audit, train, observed_ref, x_train, rep, 0)
            else:
                compare_columns(audit, train, shared, PAIR_COLUMNS, observed_ref,
                                "Same-round feature, price, group or ID pairing differs", "05")
            compare_columns(audit, truth, train, ["sample_id"], truth_ref, "Truth row is misaligned", "06")
            rates, d = draw_demand(train, rep, 0, market)
            audit.rows(np.isclose(truth["lambda"], rates, atol=1e-13, rtol=0), "03", truth_ref, truth,
                       "lambda", "Does not match the feature/price rate formula")
            audit.rows(np.isclose(truth.mean_demand, 1+rates, atol=1e-13, rtol=0), "03", truth_ref, truth,
                       "mean_demand", "Does not match 1+lambda")
            audit.rows(truth.demand.eq(d), "03", truth_ref, truth, "demand",
                       "Does not match independent seeded replay of this market")
            if market in paired_truth:
                compare_columns(audit, truth, paired_truth[market], TRUTH, truth_ref,
                                "Same-market demand and rate pairing differs", "05")
            else:
                paired_truth[market] = truth
                demand_samples[market].append(truth[["lambda", "demand"]])
            d = truth.demand.to_numpy()
            verify_sales(audit, train, d, observed_ref, "02")
            mu = np.where(train.group.eq("Low"), means.loc[(market, "Low", low/100)],
                          means.loc[(market, "High", high/100)])
            audit.rows(inventory_valid(train.inventory.to_numpy(), mu, quantile),
                       "05", observed_ref, train, "inventory", "Fails independent inventory quantile replay")
            zero_count += int(train.inventory.eq(0).sum())
            ties += int(np.sum(d == train.inventory.to_numpy()))
            if test is None or (train.inventory < 0).any():
                continue
            if train.group.nunique() != 2 or test.group.nunique() != 2:
                continue
            round_stats = {
                "scenario_id": sid, "round_id": rep, "market": market,
                "target_overall_rate": (low+high)/200, "target_low_rate": low/100, "target_high_rate": high/100,
                **coverage.add(sid, rep, train, d, test),
            }
            for label in ("overall", "low", "high"):
                rate = round_stats[f"n_censored_{label}"] / round_stats[f"n_train_{label}"]
                round_stats[f"actual_{label}_rate"] = rate
                round_stats[f"{label}_deviation_pp"] = 100 * (rate-round_stats[f"target_{label}_rate"])
            per_round.append(round_stats)
            if (sid, rep) in quality.index:
                report = quality.loc[(sid, rep)]
                expected = {
                    "market": market, "n_train": len(train), "n_test": len(test), "missing_values": 0,
                    "actual_rate": round_stats["actual_overall_rate"],
                    "zero_inventory_rate": round_stats["zero_inventory_rate_overall"],
                    "inventory_at_most_one_rate": round_stats["inventory_at_most_one_rate_overall"],
                    "train_inventory_min": round_stats["train_inventory_min_overall"],
                    "train_inventory_max": round_stats["train_inventory_max_overall"],
                    "test_inventory_min": round_stats["test_inventory_min_overall"],
                    "test_inventory_max": round_stats["test_inventory_max_overall"],
                    "test_above_train_max_rate": round_stats["test_above_pooled_train_max_rate_overall"],
                }
                for label in ("low", "high"):
                    expected.update({
                        f"n_{label}": round_stats[f"n_train_{label}"],
                        f"n_test_{label}": round_stats[f"n_test_{label}"],
                        f"actual_{label}_rate": round_stats[f"actual_{label}_rate"],
                        f"train_{label}_inventory_max": round_stats[f"train_inventory_max_{label}"],
                        f"test_{label}_above_train_max_rate": round_stats[f"test_above_group_train_max_rate_{label}"],
                    })
                for key, value in expected.items():
                    actual = report.get(key)
                    agrees = actual == value if isinstance(value, str) else (
                        actual is not None and np.isclose(actual, value, atol=1e-13, rtol=0))
                    audit.require(agrees, "04", Ref(quality_ref.file, sid, rep),
                                  "Saved quality statistic differs from audited data", field=key, value=actual)
            else:
                audit.issue("04", Ref(quality_ref.file, sid, rep), "Missing round quality record")
        if all(m in paired_truth for m in MARKETS):
            audit.require(not paired_truth[MARKETS[0]].demand.equals(paired_truth[MARKETS[1]].demand),
                          "05", Ref("ground_truth", rep=rep), "Markets unexpectedly share actual demands")
        if rep % 25 == 0:
            audit.logger.info("Scanned %d/100 scenario rounds", rep)

    audit.require(len(train_keys) == len(test_keys) == 100 and train_keys.isdisjoint(test_keys), "05", Ref("data/"),
                  "Duplicate round or train/test feature cohort")
    audit.require(len(per_round) == 1800, "04", Ref("train/"), "Incomplete round statistics")
    return pd.DataFrame(per_round), contexts, demand_samples, coverage, integrals, {"ties": ties, "zero_stock": zero_count}


def check_e0(audit, repo):
    prefix = "e0_frozen/"
    archive = audit.root / "e0_frozen"
    original = audit.root / "v1"
    original_manifest = load_manifest(original)
    e0 = open_dataset(audit.root, scenario_id="E0")
    audit.require(e0.scenarios == ("E0",) and e0.rounds == 100, "01", Ref(prefix+ID), "Incorrect E0 dataset")
    audit.require(e0.manifest["source_manifest_sha256"] == file_hash(original / "manifest.json"),
                  "09", Ref(prefix+ID), "E0 source manifest hash mismatch")
    root_paths = (archive, *archive.rglob("*"))
    for path in root_paths:
        audit.require((path.stat().st_mode & 0o222) == 0, "01", Ref(path.relative_to(audit.root).as_posix()),
                      "E0 copy must be read-only")
    root_config = repo / "config/baseline.yaml"
    audit.require(original_manifest["config_sha256"] == file_hash(root_config), "09", Ref("config/baseline.yaml"),
                  "Tracked and original E0 configuration differ")
    _, pinned = _shared_training_inputs()
    audit.require(
        pinned["source_manifest_sha256"] == file_hash(original / "manifest.json")
        and pinned["fit_seeds"] == [run["fit_seed"] for run in original_manifest["runs"]],
        "09", Ref("config/fit_seeds.json"), "Pinned fit seeds differ from the frozen manifest",
    )
    for rep in range(1, 101):
        historical_train, historical_test, historical_truth, historical_test_truth = load_run(
            original, rep - 1, include_truth=True)
        audit.file_counts["e0_original_gzip"] += 4
        audit.row_counts["e0_original_gzip"] += sum(map(len, (
            historical_train, historical_test, historical_truth, historical_test_truth)))
        refs = [
            Ref(prefix + f"train/poisson/E0/train_{rep:03d}.csv", "E0", rep),
            Ref(prefix + test_ref(rep).file, "E0", rep),
            Ref(prefix + diagnostic_ref("poisson", rep).file, "E0", rep),
            Ref(prefix + f"ground_truth/poisson/E0/truth_{rep:03d}.csv", "E0", rep),
        ]
        sources = [
            historical_train, historical_test[TEST[:-1]], historical_test,
            historical_truth[["sample_id", "true_demand", "demand_rate"]].rename(
                columns={"true_demand": "demand", "demand_rate": "lambda"}).assign(
                    mean_demand=1+historical_truth.demand_rate),
        ]
        for ref, fields, source in zip(refs, (OBSERVED, TEST, OBSERVED, TRUTH), sources):
            copied = audit.read(ref, fields, category="e0_csv_copy", e0=True)
            if copied is not None:
                compare_columns(audit, copied, source, source.columns, ref, "Archived E0 row differs", "01")
    result = subprocess.run([sys.executable, str(repo / "verify_baseline.py")], cwd=repo,
                            capture_output=True, text=True, check=False)
    audit.logger.info("E0 historic verifier: %s", (result.stdout or result.stderr).strip())
    audit.require(result.returncode == 0, "01", Ref("results/baseline-2026-11-25"),
                  "Original E0 result verification failed")
    return e0


def version_and_reproduction(audit, repo, output, config):
    current = {"python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "scipy": scipy.__version__}
    audit.require(current == config.get("versions"), "09", Ref(ID),
                  "Current Python/NumPy/Pandas/SciPy versions differ from the release", value=current)
    revision = config["code_revision"]
    for name, key in (("pricing_sim/experiment_data.py", "generator_sha256"),
                      ("pricing_sim/data_release.py", "exporter_sha256")):
        audited = subprocess.run(["git", "show", f"{revision}:{name}"], cwd=repo,
                                 capture_output=True, check=False)
        expected = config[key]
        audit.require(audited.returncode == 0 and sha256(audited.stdout).hexdigest() == expected,
                      "09", Ref(ID), f"Published source revision/hash mismatch for {name}")
        audit.require(file_hash(repo / name) == expected, "09", Ref(name),
                      "Working source differs from released generator or exporter")
    copied = 0
    chosen = ("S01", "S05", "S16", "S18")
    with tempfile.TemporaryDirectory(prefix="reproduction-", dir=output) as tmp:
        sample = Path(tmp) / "data"
        export_data(sample, rounds=2, scenarios=chosen, code_revision=revision)
        for rep in (1, 2):
            for ref in (test_ref(rep), *(diagnostic_ref(m, rep) for m in MARKETS),
                        *(f for sid in chosen for f in paths(sid, rep))):
                valid = (sample / ref.file).read_bytes() == (audit.root / ref.file).read_bytes()
                if audit.require(valid, "09", ref, "Temporary independent re-export differs byte-for-byte"):
                    copied += 1
    return {"code_revision": revision, "versions": current, "reproduced_byte_identical_files": copied,
            "scenario_ids": list(chosen), "replications": [1, 2]}

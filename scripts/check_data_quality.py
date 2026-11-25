"""Audit a complete E1–E4 data release and its separate E0 archive."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data_loader import open_dataset
from pricing_sim.learning import _q_features, _state
from qa_dataset import (
    Audit, HASH_LIST, ID, ISSUE_COLUMNS, META, Ref, check_e0, check_tree, e0_files, file_hash,
    official_file_set, paths, scan_formal_rows, verify_checksums, verify_design, version_and_reproduction,
)
from qa_model_pilot import EVIDENCE, SCENARIO_MARKETS, model_fingerprint, run_pilot
from qa_numerics import check_numeric_evaluation
from qa_statistics import EXTREME, RATE_NAMES, distribution_summary, make_figures, summarize_censoring

CHECKS = {
    "01": "目录、完整性和 E0 冻结", "02": "Schema 和逐行合法性",
    "03": "分布、固定参数和库存校准", "04": "场景与截断率",
    "05": "配对、控制变量和独立随机流", "06": "真值、共享测试集与训练隔离",
    "07": "极端场景和库存覆盖", "08": "公共评价函数独立数值核对",
    "09": "版本、完整哈希和临时复现",
    "pilot": "Naive 和 DR 真实试跑", "rsf": "RSF 整数截断边界",
    "cp": "Censored Poisson", "double_check": "报告数据一致性复核",
}
OBSERVED_FIELDS = ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta"]
MEASURED_FILES = {"train": 1800, "shared_test": 100, "training_truth": 1800, "test_diagnostics": 200}
CHECKSUM_COLUMNS = ["release", "file", "expected_sha256", "actual_sha256", "status"]


def training_contract(data):
    for sid, rep in (("S01", 1), ("S05", 50), ("S16", 100)):
        with patch.object(data, "_read", wraps=data._read) as read:
            observed = data.training(sid, rep)
        if (list(observed) != OBSERVED_FIELDS or read.call_count != 1
                or read.call_args.args[0].as_posix() != paths(sid, rep)[0].file):
            raise ValueError(f"{sid}/{rep}: training did not exclusively read the correct observation table")
        state = observed[["x1", "x2", "inventory"]].to_numpy(copy=True)
        state[:, 2] /= 10
        np.testing.assert_array_equal(_state(observed), state)
        np.testing.assert_array_equal(_q_features(observed), np.column_stack([state, observed.price]))
    return {"public_fields": OBSERVED_FIELDS, "cases": 3, "reads_training_file_only": True}


def pilot_evidence(data, destination, source=None):
    if source is None:
        evidence = run_pilot(data.root, destination)
    else:
        evidence = json.loads((source / "model_pilot.json").read_text())
    if (evidence["dataset_sha256"] != data.manifest_sha256
            or evidence["model_fingerprint"] != model_fingerprint()
            or evidence["rounds"] != 5 or evidence["scenarios"] != list(SCENARIO_MARKETS)):
        raise ValueError("Pilot evidence has a different release, fixed model, scenario or round set")
    if source is not None:
        for name in EVIDENCE:
            if file_hash(source / name) != evidence["evidence_sha256"][name]:
                raise ValueError(f"Pilot evidence checksum mismatch: {name}")
        shutil.copytree(source, destination)
    if evidence["status"] != "PASS":
        raise ValueError(f"Pilot completed {evidence['completed']}/10 scenario rounds. See model_pilot.log.")
    fits = pd.read_csv(destination / EVIDENCE[0])
    metrics = pd.read_csv(destination / EVIDENCE[1])
    methods = {"naive", "censoring_aware"}
    expected = {(sid, rep, method) for sid in SCENARIO_MARKETS for rep in range(1, 6) for method in methods}
    actual = set(fits[["scenario_id", "round_id", "method"]].itertuples(index=False, name=None))
    expected_metrics = {
        (sid, rep, method, group) for sid in SCENARIO_MARKETS for rep in range(1, 6)
        for method in methods | {"oracle"} for group in ("Overall", "Low", "High")
    }
    actual_metrics = set(metrics[["scenario_id", "round_id", "method", "group"]].itertuples(
        index=False, name=None))
    if (len(fits) != len(expected) or actual != expected or len(metrics) != len(expected_metrics)
            or actual_metrics != expected_metrics or evidence["completed"] != 10
            or not metrics.dataset_sha256.eq(data.manifest_sha256).all()
            or not np.isfinite(metrics[["profit", "profit_ratio", "regret"]]).all().all()
            or not fits.observed_rows.eq(2000).all() or not fits.reward_rows.eq(2000).all()
            or not fits.zero_inventory_rows.eq(fits.zero_sales_rows).all()
            or not fits.min_price.between(0, 1).all() or not fits.max_price.between(0, 1).all()):
        raise ValueError("Pilot files do not substantiate the claimed successful fits")
    unique = fits.drop_duplicates(["scenario_id", "round_id"])
    evidence.update({
        "verified_method_fits": len(fits), "metric_rows": len(metrics),
        "zero_rows": {sid: {"min": int(b.zero_inventory_rows.min()), "max": int(b.zero_inventory_rows.max())}
                      for sid, b in unique.groupby("scenario_id")},
        "price_min": float(fits.min_price.min()), "price_max": float(fits.max_price.max()),
        "elapsed_seconds": float(unique.elapsed_seconds.sum()),
    })
    return evidence


def largest_deviation(scenarios):
    maximum = None
    for row in scenarios.to_dict("records"):
        for group in RATE_NAMES:
            value = row[f"{group}_deviation_pp"]
            if maximum is None or abs(value) > abs(maximum["signed_pp"]):
                maximum = {"scenario_id": row["scenario_id"], "group": group, "signed_pp": float(value),
                           "target_rate": float(row[f"target_{group}_rate"]),
                           "actual_rate": float(row[f"actual_{group}_mean"])}
    return maximum


def output_double_check(audit, output, identity, e0_identity, max_deviation):
    """Reconcile saved tables independently of the in-memory summaries."""
    rounds = pd.read_csv(output / "per_round_summary.csv")
    scenarios = pd.read_csv(output / "scenario_summary.csv").set_index("scenario_id")
    coverage = pd.read_csv(output / "coverage_summary.csv")
    hashes = pd.read_csv(output / "checksum_verification.csv")
    counts = audit.file_counts
    if (len(rounds) != 1800 or len(scenarios) != 18 or
            not rounds.groupby("scenario_id").round_id.apply(lambda s: sorted(s) == list(range(1, 101))).all()
            or len(coverage) != 54 or not coverage.rounds.eq(100).all() or len(hashes) != 4312
            or hashes.file.duplicated().any() or not hashes.status.eq("PASS").all()
            or any(counts[key] != value for key, value in MEASURED_FILES.items())):
        raise ValueError("Persisted report has unexpected lengths, rounds or checksums")
    largest = 0.
    for label in RATE_NAMES:
        n = rounds[f"n_train_{label}"]
        censored = rounds[f"n_censored_{label}"]
        complete = rounds[f"n_complete_{label}"]
        np.testing.assert_array_equal(n, censored + complete)
        np.testing.assert_array_equal(n, np.full(len(rounds), 2000 if label == "overall" else 1000))
        grouped = (censored / n).groupby(rounds.scenario_id)
        for suffix, function in (("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")):
            np.testing.assert_allclose(scenarios[f"actual_{label}_{suffix}"], grouped.agg(function),
                                       atol=1e-12, rtol=0)
        deviation = 100 * (grouped.mean() - scenarios[f"target_{label}_rate"])
        np.testing.assert_allclose(scenarios[f"{label}_deviation_pp"], deviation, atol=1e-12, rtol=0)
        largest = max(largest, float(deviation.abs().max()))
    np.testing.assert_allclose(largest, abs(max_deviation["signed_pp"]), atol=1e-12, rtol=0)
    for prefix, original in (("", identity), ("e0_frozen/", e0_identity)):
        current = sha256((audit.root / (prefix + ID)).read_bytes() + b"\0"
                         + (audit.root / (prefix + HASH_LIST)).read_bytes()).hexdigest()
        if current != original:
            raise ValueError(f"Published release identity changed during the audit ({prefix or 'E1-E4'})")
    return {"scenario_rows": len(scenarios), "round_rows": len(rounds), "coverage_rows": len(coverage),
            "checksum_rows": len(hashes), "max_deviation_pp": largest}


def audit_release(data_dir, output, *, source=None, skip_pilot=False):
    logger = logging.Logger("pricing.qa.data")
    for handler in (logging.FileHandler(output / "validation.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    audit = Audit(data_dir, logger)
    statuses = {key: {"label": label, "status": "NOT TESTED"} for key, label in CHECKS.items()}
    started = datetime.now(timezone.utc).isoformat()
    logger.info("Fresh official-release audit: %s", data_dir)

    def check(keys, operation, name):
        logger.info("Begin %s – %s", "/".join(keys), name)
        try:
            result = operation()
        except Exception as error:
            logger.exception("Could not complete %s", name)
            audit.issue(keys[0], Ref(name), str(error))
            return None
        for key in keys:
            statuses[key]["status"] = "PASS"
        logger.info("Completed %s – %s", "/".join(keys), name)
        return result

    expected, expected_e0 = official_file_set(), e0_files()
    logger.info("Comparing physical file inventories with frozen design")
    check(("01",), lambda: (check_tree(audit, expected), check_tree(audit, expected_e0, "e0_frozen/")),
          "file inventory")
    identities = check(("09",), lambda: (
        verify_checksums(audit, expected | META),
        verify_checksums(audit, expected_e0 | META, "e0_frozen/"),
    ), "official and E0 checksum catalogs")
    identity, e0_identity = identities or (None, None)
    logger.info("Official release: %s; E0 CSV archive: %s", identity, e0_identity)
    config = check(("03",), lambda: verify_design(audit), ID)

    with threadpool_limits(limits=1):
        scanned = check(("02", "04", "05", "06", "07"), lambda: scan_formal_rows(audit),
                        "row schema, RNG replay, overlaps, coverage and censoring for all 3900 CSVs")
        e0 = check(("01",), lambda: check_e0(audit, ROOT), "frozen E0 and original results")
        data = check(("06",), lambda: open_dataset(data_dir), "public loader")
        if data is not None:
            audit.require(data.manifest_sha256 == identity, "09", Ref(ID), "Public loader uses a different release")
        if e0 is not None:
            audit.require(e0.manifest_sha256 == e0_identity, "09", Ref("e0_frozen/" + ID),
                          "Public loader uses a different E0 release")
        loader = check(("06",), lambda: training_contract(data), "seven-field training input contract") if data else None
        numerical = check(("08",), lambda: check_numeric_evaluation(data, e0),
                          "independent discrete probability sums") if data and e0 else None
        if numerical:
            for edge in numerical["off_release_warnings"]:
                audit.issue("08", Ref("pricing_sim/experiment_data.py"),
                            edge["reason"], field="conditional_censored_demand",
                            value=json.dumps(edge, ensure_ascii=False), severity="WARNING")
        provenance = check(("09",), lambda: version_and_reproduction(audit, ROOT, output, config),
                           "version provenance and temporary reproduction") if config else None

    per_round = scenarios = distribution = coverage_summary = pd.DataFrame()
    row_details, figures, maximum = {}, [], None
    if scanned:
        per_round, contexts, demand, coverage, calibration, row_details = scanned
        scenarios = summarize_censoring(per_round)
        coverage_summary, inventory = coverage.summary()
        distribution = distribution_summary(contexts, demand, inventory, calibration)
        maximum = largest_deviation(scenarios)
        if maximum is not None:
            logger.info("Largest 100-round censoring deviation: %s %s %+.4f pp",
                        maximum["scenario_id"], maximum["group"], maximum["signed_pp"])
            for row in scenarios.to_dict("records"):
                sid = row["scenario_id"]
                for label in ("low", "high"):
                    value = row[f"{label}_deviation_pp"]
                    audit.require(abs(value) <= 1, "04", paths(sid, 1)[0],
                                  "100-round censoring deviation exceeds 1 pp", field=label, value=value)
        for row in coverage_summary.to_dict("records"):
            if row["scenario_id"] in EXTREME and row["group"] == "Overall":
                sid = row["scenario_id"]
                audit.issue("07", Ref(str(Path(paths(sid, 1)[0].file).parent), sid),
                            "Legal 90% target group has limited inventory/demand information",
                            field="zero_inventory_rate_overall", value=row["zero_inventory_rate"], severity="WARNING")
        result = check(("07",), lambda: make_figures(output / "figures", per_round, coverage, coverage_summary),
                       "coverage and inventory figures")
        figures = result or []
    for key, planned in MEASURED_FILES.items():
        ref = Ref(key)
        audit.require(audit.file_counts[key] == planned and audit.row_counts[key] == 2000 * planned,
                      "01", ref, "File or row counts differ from the full official release")

    pilot = None
    if not skip_pilot and data is not None:
        pilot = check(("pilot",), lambda: pilot_evidence(data, output / "model_pilot", source), "real model fits")
    if pilot:
        boundary = pilot["rsf_discrete_boundary"]
        statuses["rsf"]["status"] = boundary["status"]
        if boundary["status"] == "WARNING":
            audit.issue("rsf", Ref("pricing_sim/survival.py"),
                        "Discrete censor boundary needs a risk-set and tail review before release",
                        field="E[D|D>=Y] (D=1+Poisson(1), Y=2)",
                        value=f"{boundary['current']:.8f} vs {boundary['correct_conditional_mean']:.8f}",
                        severity="WARNING")
        logger.info("Confirmed %d/10 pilot cases, %d fitted methods; RSF %s; Censored Poisson NOT TESTED",
                    pilot["completed"], pilot["verified_method_fits"], boundary["status"])

    for name, frame in (("per_round_summary", per_round), ("scenario_summary", scenarios),
                        ("coverage_summary", coverage_summary), ("distribution_summary", distribution),
                        ("checksum_verification", pd.DataFrame(audit.checksums, columns=CHECKSUM_COLUMNS))):
        frame.to_csv(output / f"{name}.csv", index=False)
    double = check(("double_check",), lambda: output_double_check(audit, output, identity, e0_identity, maximum),
                   "saved counts, recomputed censoring and release IDs") if maximum is not None else None
    for issue in audit.issues:
        status = statuses[issue["check"]]
        if issue["severity"] == "FAIL" or status["status"] == "PASS":
            status["status"] = issue["severity"]
    dataset_statuses = [statuses[f"{number:02d}"]["status"] for number in range(1, 10)]
    dataset_ready = (all(x in {"PASS", "WARNING"} for x in dataset_statuses)
                     and statuses["double_check"]["status"] == "PASS")
    pd.DataFrame(audit.issues, columns=ISSUE_COLUMNS).to_csv(output / "anomalies.csv", index=False)
    summary = {
        "data_path": str(data_dir), "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": identity, "e0_archive_sha256": e0_identity, "checks": statuses,
        "official_files": sum(audit.file_counts[k] for k in MEASURED_FILES),
        "official_physical_rows": sum(audit.row_counts[k] for k in MEASURED_FILES),
        "official_main_rows": audit.row_counts["train"] + audit.row_counts["shared_test"],
        "files_by_type": dict(audit.file_counts), "rows_by_type": dict(audit.row_counts),
        "checksums_verified": len(audit.checksums),
        "checksums_passed": sum(row["status"] == "PASS" for row in audit.checksums),
        "official_row_details": row_details, "max_censoring_deviation": maximum,
        "issue_count": len(audit.issues), "failed_issues": sum(x["severity"] == "FAIL" for x in audit.issues),
        "warned_issues": sum(x["severity"] == "WARNING" for x in audit.issues),
        "training_contract": loader, "numeric_evaluation": numerical, "provenance": provenance,
        "model_pilot": pilot, "figures": figures, "double_check": double,
        "dataset_release_readiness": "READY" if dataset_ready else "NOT READY",
        "model_experiment_readiness": "PENDING" if statuses["pilot"]["status"] == "PASS" else "NOT VERIFIED",
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    from qa_report import write_report
    write_report(output, summary, scenarios, distribution, coverage_summary)
    logger.info("Summary verified: official files=%d, main rows=%d, physical rows=%d, hashes=%d/%d, "
                "warnings=%d, failures=%d; dataset=%s, models=%s",
                summary["official_files"], summary["official_main_rows"], summary["official_physical_rows"],
                summary["checksums_passed"], summary["checksums_verified"], summary["warned_issues"],
                summary["failed_issues"], summary["dataset_release_readiness"], summary["model_experiment_readiness"])
    return dataset_ready and statuses["pilot"]["status"] != "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/data_quality")
    pilot = parser.add_mutually_exclusive_group()
    pilot.add_argument("--skip-model-pilot", action="store_true")
    pilot.add_argument("--pilot-from", type=Path, help="Attach verified pilot evidence for this exact release/model")
    args = parser.parse_args()
    data_dir, output = args.data.resolve(), args.output.resolve()
    if output.is_relative_to(data_dir):
        parser.error("--output must be outside the published data directory")
    if output.exists():
        parser.error("--output already exists; choose a new audit output directory")
    output.mkdir(parents=True)
    return 0 if audit_release(data_dir, output, source=args.pilot_from, skip_pilot=args.skip_model_pilot) else 1


if __name__ == "__main__":
    sys.exit(main())

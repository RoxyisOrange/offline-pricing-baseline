"""Train the three baseline policies on frozen observed datasets."""
import argparse
import json
from pathlib import Path

import pandas as pd

from pricing_sim.data_store import load_config, load_manifest, load_run, sha256
from pricing_sim.evaluation import evaluate_prices, per_run_metrics, summarize
from pricing_sim.experiment_data import SCENARIOS, ExperimentData
from pricing_sim.experiment_evaluation import evaluate_fitted_models, summarize_metrics
from pricing_sim.learning import OraclePricing, PricingPipeline

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "results" / "baseline-2026-11-25"


def train_shared(data_dir, output, scenarios, limit, resume):
    data = ExperimentData(data_dir)
    count = data.rounds if limit is None else limit
    if not 1 <= count <= data.rounds:
        raise ValueError("--limit outside exported round count")
    if len(set(scenarios)) != len(scenarios) or set(scenarios).difference(data.scenarios):
        raise ValueError("Choose distinct scenarios present in the shared export")

    legacy_dir = ROOT / "data" / "v1"
    legacy = load_manifest(legacy_dir, verify_hashes=False)
    config = load_config(legacy_dir / "config.yaml")
    manifest = {
        "dataset_sha256": data.manifest_sha256,
        "generator_version": data.manifest["version"],
        "config_sha256": legacy["config_sha256"],
        "fit_seed_manifest_sha256": sha256(legacy_dir / "manifest.json"),
        "scenarios": scenarios, "rounds": count, "methods": ["naive", "censoring_aware"],
    }
    manifest_path = output / "run_manifest.json"
    if resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Cannot resume with a different or missing run manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    metrics = []
    for round_id in range(1, count + 1):
        cache = {}
        seed = legacy["runs"][round_id - 1]["fit_seed"]
        for scenario in scenarios:
            path = checkpoints / f"{scenario}_{round_id:03d}.csv"
            if resume and path.exists():
                score = pd.read_csv(path, float_precision="round_trip")
            else:
                train = data.training(scenario, round_id)
                naive = PricingPipeline(config, "naive", seed).fit(train)
                aware = PricingPipeline(config, "censoring_aware", seed).fit(
                    train, density_cache=naive.density_cache)
                score = evaluate_fitted_models(
                    data, scenario, round_id, {"naive": naive, "censoring_aware": aware}, cache=cache)
                temporary = path.with_suffix(".tmp")
                score.to_csv(temporary, index=False)
                temporary.replace(path)
            metrics.append(score)
            print(f"{scenario} round {round_id}/{count} complete", flush=True)
    combined = pd.concat(metrics, ignore_index=True)
    combined.to_csv(output / "metrics_per_run.csv", index=False)
    summarize_metrics(combined).to_csv(output / "metrics_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Frozen data directory (E0 defaults to data/v1)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS),
                        help="Run selected E1–E4 scenarios from a shared export")
    args = parser.parse_args()
    if args.scenarios and args.data is None:
        parser.error("--scenarios requires --data pointing to a shared export")
    data_dir = (args.data or ROOT / "data" / "v1").resolve()
    output = args.output.resolve()
    if output == BASELINE.resolve():
        raise ValueError("The archived baseline must not be overwritten")
    if args.scenarios:
        train_shared(data_dir, output, args.scenarios, args.limit, args.resume)
        return
    manifest = load_manifest(data_dir)
    config = load_config(data_dir / "config.yaml")
    count = manifest["n_repeats"] if args.limit is None else args.limit
    if not 1 <= count <= manifest["n_repeats"]:
        raise ValueError("--limit outside frozen repeat count")
    run_info = {"data_manifest_sha256": sha256(data_dir / "manifest.json"),
                "config_sha256": manifest["config_sha256"], "n_repeats": count,
                "method": "python_survival_approximation"}
    run_manifest = output / "run_manifest.json"
    if args.resume:
        if not run_manifest.exists() or json.loads(run_manifest.read_text()) != run_info:
            raise ValueError("Cannot resume with a different or missing run manifest")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        run_manifest.write_text(json.dumps(run_info, indent=2) + "\n")
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    all_predictions, all_metrics, diagnostics = [], [], []
    for run in range(count):
        prefix = checkpoints / f"run_{run:03d}"
        pred_file = Path(str(prefix) + "_predictions.csv.gz")
        metric_file = Path(str(prefix) + "_metrics.csv")
        diag_file = Path(str(prefix) + "_diagnostics.json")
        if args.resume and all(p.exists() for p in (pred_file, metric_file, diag_file)):
            pred = pd.read_csv(pred_file, compression="gzip", float_precision="round_trip")
            rows = pd.read_csv(metric_file).to_dict("records")
            diag = json.loads(diag_file.read_text())
        else:
            train, test = load_run(data_dir, run)
            seed = manifest["runs"][run]["fit_seed"]
            oracle = OraclePricing(config).fit(train)
            naive = PricingPipeline(config, "naive", seed).fit(train)
            aware = PricingPipeline(config, "censoring_aware", seed).fit(
                train, density_cache=naive.density_cache)
            x1, x2, inventory = (test[k].to_numpy() for k in ("x1", "x2", "inventory"))
            pred = evaluate_prices(test, {
                "oracle": oracle.predict_price(x1, x2, inventory),
                "naive": naive.predict_price(x1, x2, inventory),
                "censoring_aware": aware.predict_price(x1, x2, inventory)}, config, run)
            rows = per_run_metrics(pred, run)
            diag = {"run": run, "naive": naive.diagnostics, "censoring_aware": aware.diagnostics}
            pred.to_csv(pred_file, index=False, compression="gzip")
            pd.DataFrame(rows).to_csv(metric_file, index=False)
            diag_file.write_text(json.dumps(diag, indent=2) + "\n")
        all_predictions.append(pred)
        all_metrics.extend(rows)
        diagnostics.append(diag)
        print(f"run {run+1}/{count} complete", flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions.csv.gz",
                                                       index=False, compression="gzip")
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(output / "metrics_per_run.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(summary.query("metric == 'performance_ratio'")["method value".split()].to_string(index=False))


if __name__ == "__main__":
    main()

"""Train the three baseline policies on frozen observed datasets."""
import argparse
import json
from pathlib import Path

import pandas as pd

from pricing_sim.data_store import load_config, load_manifest, load_run, sha256
from pricing_sim.evaluation import evaluate_prices, per_run_metrics, summarize
from pricing_sim.learning import OraclePricing, PricingPipeline

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "results" / "baseline-2026-11-25"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data_dir, output = args.data.resolve(), args.output.resolve()
    if output == BASELINE.resolve():
        raise ValueError("The archived baseline must not be overwritten")
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

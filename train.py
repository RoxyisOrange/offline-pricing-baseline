"""Train selected scenarios using the shared release, or replay archived E0."""
import argparse
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path

import pandas as pd

from data_loader import get_scenarios, open_dataset
from pricing_sim.data_store import load_config, load_manifest, load_run, sha256
from pricing_sim.evaluation import evaluate_prices, per_run_metrics, summarize
from pricing_sim.experiment_data import EXPERIMENTS, SCENARIOS
from pricing_sim.experiment_evaluation import evaluate_fitted_models, summarize_metrics
from pricing_sim.learning import OraclePricing, PricingPipeline

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "results" / "baseline-2026-11-25"


def _shared_training_inputs():
    config_dir = ROOT / "config"
    config_path = config_dir / "baseline.yaml"
    pinned = json.loads((config_dir / "fit_seeds.json").read_text(encoding="utf-8"))
    if sha256(config_path) != pinned["config_sha256"]:
        raise ValueError("Shared training configuration differs from the pinned baseline")
    config = load_config(config_path)
    seeds = pinned["fit_seeds"]
    serialized = json.dumps(seeds, separators=(",", ":")).encode()
    if (hash_bytes(serialized).hexdigest() != pinned["fit_seeds_sha256"]
            or len(seeds) != config["simulation"]["n_repeats"]
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Invalid pinned fit seeds")
    return config, pinned


def train_shared(data_dir, output, scenarios, limit, resume):
    data = open_dataset(data_dir)
    count = data.rounds if limit is None else limit
    if not 1 <= count <= data.rounds:
        raise ValueError("--limit outside exported round count")
    if not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios).difference(data.scenarios):
        raise ValueError("Choose distinct scenarios present in the shared export")

    config, pinned = _shared_training_inputs()
    if count > len(pinned["fit_seeds"]):
        raise ValueError("Requested rounds exceed the pinned fit seeds")
    manifest = {
        "dataset_sha256": data.manifest_sha256,
        "generator_version": data.manifest["version"],
        "config_sha256": pinned["config_sha256"],
        "fit_seed_manifest_sha256": pinned["source_manifest_sha256"],
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
        seed = pinned["fit_seeds"][round_id - 1]
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


def replay_e0(data_dir, output, limit, resume):
    if (data_dir / "manifest.json").is_file():
        # Explicit replay also supports the original E0 bundle.
        manifest = load_manifest(data_dir)
        config = load_config(data_dir / "config.yaml")
        seeds = [row["fit_seed"] for row in manifest["runs"]]
        config_hash = manifest["config_sha256"]
        data_hash = sha256(data_dir / "manifest.json")

        def observed_pair(run):
            return load_run(data_dir, run)
    else:
        archive = open_dataset(data_dir, scenario_id="E0")
        if archive.scenarios != ("E0",):
            raise ValueError("--replay-e0 requires the E0 archive")
        config = archive.manifest["legacy_config"]
        seeds = [row["fit_seed"] for row in archive.manifest["source_runs"]]
        if len(seeds) != archive.rounds:
            raise ValueError("Incomplete E0 fit seed list")
        config_hash = hash_bytes(json.dumps(config, sort_keys=True).encode()).hexdigest()
        data_hash = archive.manifest_sha256

        def observed_pair(run):
            return archive.training("E0", run + 1), archive.test(run + 1)

    count = len(seeds) if limit is None else limit
    if not 1 <= count <= len(seeds):
        raise ValueError("--limit outside archived round count")
    run_info = {"data_manifest_sha256": data_hash,
                "config_sha256": config_hash, "n_repeats": count,
                "method": "python_survival_approximation"}
    run_manifest = output / "run_manifest.json"
    if resume:
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
        if resume and all(p.exists() for p in (pred_file, metric_file, diag_file)):
            pred = pd.read_csv(pred_file, compression="gzip", float_precision="round_trip")
            rows = pd.read_csv(metric_file).to_dict("records")
            diag = json.loads(diag_file.read_text())
        else:
            train, test = observed_pair(run)
            seed = seeds[run]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="CSV release root (default: data/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS),
                           help="Train selected E1–E4 scenario IDs")
    selection.add_argument("--experiment", choices=tuple(EXPERIMENTS),
                           help="Train the scenarios in an experiment")
    selection.add_argument("--replay-e0", action="store_true",
                           help="Explicitly replay E0 from data/e0_frozen")
    args = parser.parse_args()
    data_dir, output = args.data.resolve(), args.output.resolve()
    if output == BASELINE.resolve():
        raise ValueError("The archived baseline must not be overwritten")
    if args.replay_e0:
        replay_e0(data_dir, output, args.limit, args.resume)
    else:
        scenarios = args.scenarios or get_scenarios(args.experiment)
        train_shared(data_dir, output, scenarios, args.limit, args.resume)


if __name__ == "__main__":
    main()

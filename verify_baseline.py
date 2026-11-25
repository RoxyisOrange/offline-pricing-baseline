"""Check frozen data and the archived 100-run results without retraining."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_sim.data_store import load_config, load_manifest, load_run, sha256
from pricing_sim.evaluation import METHODS, METRICS, per_run_metrics, summarize
from pricing_sim.simulation import expected_profit

ROOT = Path(__file__).resolve().parent


def main():
    data_dir = ROOT / "data" / "v1"
    result_dir = ROOT / "results" / "baseline-2026-11-25"
    manifest = load_manifest(data_dir)
    if sha256(ROOT / "config" / "baseline.yaml") != manifest["config_sha256"]:
        raise ValueError("Root configuration differs from frozen version")
    provenance = json.loads((result_dir / "provenance.json").read_text())
    if sha256(data_dir / "manifest.json") != provenance["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from result provenance")
    for name, digest in provenance["result_sha256"].items():
        if sha256(result_dir / name) != digest:
            raise ValueError(f"Result changed: {name}")
    config = load_config(data_dir / "config.yaml")
    predictions = pd.read_csv(result_dir / "predictions.csv.gz", float_precision="round_trip")
    saved_metrics = pd.read_csv(result_dir / "metrics_per_run.csv", float_precision="round_trip")
    saved_summary = pd.read_csv(result_dir / "metrics_summary.csv", float_precision="round_trip")
    if len(predictions) != 200000 or predictions.run.nunique() != 100:
        raise ValueError("Expected 100 runs and 200,000 predictions")
    rebuilt = []
    for run, group in predictions.groupby("run", sort=True):
        _, observed_test = load_run(data_dir, int(run))
        group = group.sort_values("sample_id").reset_index(drop=True)
        for name in ("sample_id", "x1", "x2", "inventory"):
            np.testing.assert_allclose(group[name], observed_test[name], rtol=0, atol=2e-15)
        for method in METHODS:
            profit = expected_profit(group.x1.to_numpy(), group.x2.to_numpy(),
                                     group.inventory.to_numpy(),
                                     group[f"price_{method}"].to_numpy(), config)
            np.testing.assert_allclose(group[f"profit_{method}"], profit, rtol=0, atol=2e-12)
        rebuilt.extend(per_run_metrics(group, int(run)))
    rebuilt = pd.DataFrame(rebuilt)
    for metric in METRICS:
        np.testing.assert_allclose(rebuilt[metric], saved_metrics[metric], rtol=0, atol=2e-12)
    summary = summarize(rebuilt)
    for name in ("value", "standard_error", "ci95_low", "ci95_high"):
        np.testing.assert_allclose(summary[name], saved_summary[name], rtol=0, atol=2e-12)
    ratios = saved_summary.query("metric == 'performance_ratio'").set_index("method").value
    print(f"Verified 100 data pairs and 200,000 predictions; Naive {ratios.naive:.6%}, "
          f"DR {ratios.censoring_aware:.6%}")


if __name__ == "__main__":
    main()

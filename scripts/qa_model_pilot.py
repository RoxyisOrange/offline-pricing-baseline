"""Run the real Naive/DR models on a small, separate QA pilot."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import poisson
from threadpoolctl import threadpool_limits

from data_loader import open_dataset
from pricing_sim.experiment_evaluation import Predictions, evaluate_round
from pricing_sim.learning import PricingPipeline
from pricing_sim.survival import surrogate_reward
from train import _shared_training_inputs

SCENARIO_MARKETS = {"S05": "poisson", "S16": "negative_binomial"}
EVIDENCE = ("model_pilot_per_run.csv", "model_pilot_metrics.csv", "model_pilot.log")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint():
    paths = (
        "config/baseline.yaml", "config/fit_seeds.json", "data_loader.py", "train.py",
        "pricing_sim/learning.py", "pricing_sim/survival.py", "pricing_sim/experiment_data.py",
        "pricing_sim/data_release.py", "pricing_sim/experiment_evaluation.py",
    )
    return sha256("\n".join(f"{name} {digest(ROOT / name)}" for name in paths).encode()).hexdigest()


def discrete_boundary():
    """Exercise the production reward function with an exactly known survival curve."""
    class ExactForest:
        max_time = 48

        def survival(self, features):
            curve = poisson.sf(np.arange(self.max_time + 1) - 1, 1.0)
            return np.repeat(curve[None, :], len(features), axis=0)

    # D = 1 + Poisson(1), Y=2. The recovery target includes D=Y.
    observed = pd.DataFrame({"x1": [0.0], "x2": [0.0], "price": [0.8],
                             "inventory": [2], "sales": [2], "delta": [0]})
    reward, _ = surrogate_reward(observed, ExactForest(), .1)
    current = float(2 + (1.6 - reward[0]) / .1)
    k = np.arange(256)
    pmf = poisson.pmf(k, 1)
    truth = float(np.dot(1 + k[k >= 1], pmf[k >= 1]) / pmf[k >= 1].sum())
    return {
        "status": "WARNING" if not np.isclose(current, truth, atol=1e-12) else "PASS",
        "current": current, "correct_conditional_mean": truth,
        "description": "H(Y) denominator, Y in the risk set at a same-time event, "
                       "and tail summation through the observed sales cap require a discrete review.",
    }


def check_policy(model, observed, test, diagnostic, method):
    n = len(observed)
    for field in ("reward", "q_logged", "q_smooth", "residual", "inverse_density"):
        values = np.asarray(getattr(model, field))
        if values.shape[0] != n or not np.isfinite(values).all():
            raise ValueError(f"{method}: nonfinite or missing rows in {field}")
    if method == "naive":
        np.testing.assert_array_equal(model.reward, observed.price * observed.sales)
    else:
        forest = model._demand_forest
        np.testing.assert_array_equal(forest.features, observed[["x1", "x2", "price"]].to_numpy())
        np.testing.assert_array_equal(forest.times, observed.sales)
        np.testing.assert_array_equal(forest.events, observed.delta)
        if model.diagnostics["survival"]["censored"] != (observed.delta == 0).sum():
            raise ValueError("Survival fit did not account for all censored rows")

    state = [test[column].to_numpy() for column in ("x1", "x2", "inventory")]
    prices = model.predict_price(*state)
    recovery = model.predict_censored_demand(diagnostic)
    q = model.predict_q(diagnostic)
    if (prices.shape != (len(test),) or not np.isfinite(prices).all()
            or (prices < 0).any() or (prices > 1).any()):
        raise ValueError(f"{method}: invalid test prices")
    if not np.isfinite(recovery).all() or not np.isfinite(q).all():
        raise ValueError(f"{method}: nonfinite diagnostic predictions")
    return Predictions(prices, recovery, q)


def run_pilot(data_dir, output, rounds=5):
    output = Path(output)
    data_dir = Path(data_dir).resolve()
    if output.resolve().is_relative_to(data_dir):
        raise ValueError("Pilot output must be outside the data directory")
    data = open_dataset(data_dir)
    config, pinned = _shared_training_inputs()
    if not 1 <= rounds <= min(data.rounds, len(pinned["fit_seeds"])):
        raise ValueError("Pilot round count outside the data/seed range")
    output.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("pricing.qa.model")
    handlers = [logging.FileHandler(output / "model_pilot.log", encoding="utf-8"),
                logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.info("Real fits for %s, rounds 1–%s, data=%s", tuple(SCENARIO_MARKETS),
                rounds, data.manifest_sha256)
    rows, metrics, failures = [], [], []
    with threadpool_limits(limits=1):
        for rep in range(1, rounds + 1):
            oracle_cache = {}
            for sid, market in SCENARIO_MARKETS.items():
                start = perf_counter()
                logger.info("%s round %03d starting", sid, rep)
                try:
                    actions = data.training(sid, rep)
                    if list(actions) != ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta"]:
                        raise ValueError("Unexpected training fields")
                    seed = pinned["fit_seeds"][rep - 1]
                    with np.errstate(divide="raise", invalid="raise", over="raise"):
                        naive = PricingPipeline(config, "naive", seed).fit(actions)
                        aware = PricingPipeline(config, "censoring_aware", seed).fit(
                            actions, density_cache=naive.density_cache)
                        test = data.test(rep)
                        diagnostic = data.diagnostics(rep, market).loc[:, actions.columns]
                        predictions = {
                            method: check_policy(model, actions, test, diagnostic, method)
                            for method, model in (("naive", naive), ("censoring_aware", aware))
                        }
                        score = evaluate_round(data, sid, rep, predictions, cache=oracle_cache)
                    if not np.isfinite(score[["profit", "regret", "profit_ratio"]]).all().all():
                        raise ValueError("Nonfinite profit, regret or performance ratio")
                    metrics.append(score)
                    for method, prediction in predictions.items():
                        prices = np.asarray(prediction.price)
                        rows.append({
                            "scenario_id": sid, "round_id": rep, "method": method, "fit_seed": seed,
                            "observed_rows": len(actions), "reward_rows": len(actions),
                            "zero_inventory_rows": int(actions.inventory.eq(0).sum()),
                            "zero_sales_rows": int(actions.sales.eq(0).sum()),
                            "min_price": prices.min(), "max_price": prices.max(),
                            "elapsed_seconds": round(perf_counter() - start, 3),
                        })
                    logger.info("%s round %03d passed, %d rows, zero-inventory=%d, %.2fs",
                                sid, rep, len(actions), actions.inventory.eq(0).sum(), perf_counter() - start)
                except Exception as error:
                    logger.exception("%s round %03d failed", sid, rep)
                    failures.append({"scenario_id": sid, "round_id": rep, "reason": str(error)})

    boundary = discrete_boundary()
    logger.info("Boundary probe: current=%.8f, E[D|D>=Y]=%.8f", boundary["current"],
                boundary["correct_conditional_mean"])
    logger.info("Finished %d/%d scenario rounds; CP NOT TESTED", len(metrics), 2 * rounds)
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    pd.DataFrame(rows).to_csv(output / EVIDENCE[0], index=False)
    (pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()).to_csv(
        output / EVIDENCE[1], index=False)
    result = {
        "dataset_sha256": data.manifest_sha256, "model_fingerprint": model_fingerprint(),
        "rounds": rounds, "scenarios": list(SCENARIO_MARKETS),
        "completed": len(metrics), "failures": failures,
        "status": "PASS" if len(metrics) == len(SCENARIO_MARKETS) * rounds else "FAIL",
        "rsf_discrete_boundary": boundary,
        "censored_poisson": {"status": "NOT TESTED", "reason": "No fitted CP implementation in this repo"},
        "evidence_sha256": {name: digest(output / name) for name in EVIDENCE},
    }
    (output / "model_pilot.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    sys.exit(0 if run_pilot(args.data, args.output, args.rounds)["status"] == "PASS" else 1)

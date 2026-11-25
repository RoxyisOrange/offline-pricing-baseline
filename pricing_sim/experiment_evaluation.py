"""Shared evaluation of Poisson and NB policies on the frozen test rounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import t

from .experiment_data import (GROUPS, ORACLE_STEP, SCENARIOS, TEST_COLUMNS, VERSION,
                              ExperimentData, conditional_censored_demand, expected_profit)
from .simulation import OBSERVED_COLUMNS

METRICS = ("profit", "profit_ratio", "regret", "correction_value",
           "price_mae", "price_bias", "demand_mae", "q_mae")


@dataclass(frozen=True)
class Predictions:
    price: ArrayLike
    censored_mean: ArrayLike | None = None
    q_logged: ArrayLike | None = None


def _columns(frame, names):
    return tuple(frame[name].to_numpy() for name in names)


def _oracle(test, market):
    grid = np.linspace(0, 1, round(1 / ORACLE_STEP) + 1)
    prices = np.empty(len(test))
    profits = np.empty(len(test))
    for start in range(0, len(test), 128):
        block = slice(start, start + 128)
        x1, x2, inventory = _columns(test.iloc[block], ("x1", "x2", "inventory"))
        q = expected_profit(x1[:, None], x2[:, None], inventory[:, None], grid[None, :], market)
        best = q.argmax(axis=1)
        prices[block] = grid[best]
        profits[block] = q[np.arange(len(best)), best]
    return prices, profits


@dataclass(frozen=True)
class _Targets:
    test: pd.DataFrame
    oracle_price: np.ndarray
    oracle_profit: np.ndarray
    censored: np.ndarray
    observed_sales: np.ndarray
    censored_mean: np.ndarray
    q_logged: np.ndarray


def _targets(data, round_id, market):
    test = data.test(round_id)
    diag = data.diagnostics(round_id, market)
    if not test.equals(diag[list(TEST_COLUMNS)]):
        raise ValueError("Evaluation observations do not align with the shared test set")
    x1, x2, inventory, price = _columns(diag, ("x1", "x2", "inventory", "price"))
    censored = diag.delta.to_numpy() == 0
    conditional_mean = conditional_censored_demand(
        x1[censored], x2[censored], inventory[censored], price[censored], market)
    oracle_price, oracle_profit = _oracle(test, market)
    return _Targets(test, oracle_price, oracle_profit, censored, diag.sales.to_numpy(),
                    conditional_mean, expected_profit(x1, x2, inventory, price, market))


def _vector(values, length, name):
    array = np.asarray(values, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must provide {length} finite values in test row order")
    return array


def evaluate_round(data: ExperimentData, scenario_id: str, round_id: int,
                   predictions: dict[str, Predictions], *, cache: dict | None = None) -> pd.DataFrame:
    """Return tidy metrics for one scenario and round. `profit_ratio` is a fraction.

    Price and Q arrays follow the shared test order. Censored demand predictions
    follow the order of rows with delta=0 in the market's diagnostic table.
    Supply the same cache across scenarios to reuse that round's market Oracle.
    """
    if not predictions or "oracle" in predictions:
        raise ValueError("Provide learned-method predictions. Oracle is computed here.")
    market = SCENARIOS[scenario_id].market
    key = (data.manifest_sha256, round_id, market)
    if cache is None:
        cache = {}
    if key not in cache:
        cache[key] = _targets(data, round_id, market)
    target = cache[key]
    test = target.test
    x1, x2, inventory = _columns(test, ("x1", "x2", "inventory"))
    rates = data.rates(scenario_id, round_id)

    prices = {"oracle": target.oracle_price}
    profits = {"oracle": target.oracle_profit}
    demand_errors, q_errors = {}, {}
    for method, supplied in predictions.items():
        price = _vector(supplied.price, len(test), f"{method} price")
        prices[method] = price
        profits[method] = expected_profit(x1, x2, inventory, price, market)
        demand = supplied.censored_mean
        if demand is None and method == "naive":
            demand = target.observed_sales[target.censored]
        if demand is not None:
            estimate = _vector(demand, len(target.censored_mean), f"{method} censored mean")
            demand_errors[method] = np.abs(estimate - target.censored_mean)
        if supplied.q_logged is not None:
            q = _vector(supplied.q_logged, len(test), f"{method} Q")
            q_errors[method] = np.abs(q - target.q_logged)

    groups = test.group.to_numpy()
    rows = []
    for group in ("Overall", *GROUPS):
        subset = np.ones(len(test), dtype=bool) if group == "Overall" else groups == group
        censored_subset = subset[target.censored]
        oracle_profit = float(target.oracle_profit[subset].mean())
        for method, values in profits.items():
            profit = float(values[subset].mean())
            price_error = (prices[method] - target.oracle_price)[subset]
            rows.append({
                "scenario_id": scenario_id, "market": market, "round_id": round_id,
                "generator_version": VERSION, "dataset_sha256": data.manifest_sha256,
                "group": group, "method": method, **rates,
                "n_test": int(subset.sum()), "n_censored_diagnostic": int(censored_subset.sum()),
                "profit": profit, "oracle_profit": oracle_profit,
                "profit_ratio": profit / oracle_profit if oracle_profit > 0 else np.nan,
                "regret": oracle_profit - profit,
                "correction_value": profit - float(profits["naive"][subset].mean())
                if "naive" in profits and method not in ("oracle", "naive") else np.nan,
                "price_mae": float(np.abs(price_error).mean()),
                "price_bias": float(price_error.mean()),
                "demand_mae": float(demand_errors[method][censored_subset].mean())
                if method in demand_errors and censored_subset.any() else np.nan,
                "q_mae": float(q_errors[method][subset].mean()) if method in q_errors else np.nan,
            })
    return pd.DataFrame(rows)


def evaluate_fitted_models(data: ExperimentData, scenario_id: str, round_id: int,
                           models: dict, *, cache: dict | None = None) -> pd.DataFrame:
    """Score fitted E0 pipelines, including recovery error and Q error."""
    market = SCENARIOS[scenario_id].market
    test = data.test(round_id)
    diagnostic = data.diagnostics(round_id, market)[OBSERVED_COLUMNS]
    state = _columns(test, ("x1", "x2", "inventory"))
    predictions = {
        name: Predictions(model.predict_price(*state), model.predict_censored_demand(diagnostic),
                          model.predict_q(diagnostic))
        for name, model in models.items()
    }
    return evaluate_round(data, scenario_id, round_id, predictions, cache=cache)


def _estimate(value, samples):
    n = len(samples)
    se = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    width = t.ppf(0.975, n - 1) * se if n > 1 else np.nan
    return {"value": value, "standard_error": se, "ci95_low": value - width,
            "ci95_high": value + width, "n_repeats": n}


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equally across rounds. The overall ratio is Eq. (23)."""
    keys = ["scenario_id", "market", "generator_version", "dataset_sha256", "group", "method"]
    rows = []
    for label, block in metrics.groupby(keys, sort=False):
        if block.round_id.duplicated().any():
            raise ValueError("Duplicate result for a scenario, round, group and method")
        for metric in METRICS:
            if metric == "profit_ratio":
                profit = block.profit.to_numpy()
                oracle = block.oracle_profit.to_numpy()
                denominator = oracle.mean()
                value = float(profit.mean() / denominator) if denominator > 0 else np.nan
                # Delta method applied to the same paired rounds in numerator and denominator.
                samples = (profit - value * oracle) / denominator if denominator > 0 else np.array([])
            else:
                samples = block[metric].dropna().to_numpy()
                value = float(samples.mean()) if len(samples) else np.nan
            rows.append({**dict(zip(keys, label)), "metric": metric, **_estimate(value, samples)})
    return pd.DataFrame(rows)


def paired_contrast(metrics: pd.DataFrame, weights: dict[tuple[str, str], float],
                    *, metric: str = "profit", group: str = "Overall") -> dict:
    """Compute a weighted difference of scenario/method means, paired by round."""
    if not weights or metric not in METRICS or metric == "profit_ratio":
        raise ValueError("Choose a non-ratio metric and at least one scenario/method weight")
    selected = metrics.loc[metrics.group == group].set_index(["scenario_id", "method"])
    columns = []
    for key in weights:
        series = selected.loc[[key]].set_index(["dataset_sha256", "round_id"])[metric]
        if not series.index.is_unique:
            raise ValueError("Select exactly one result per method and round")
        columns.append(series)
    paired = pd.concat(columns, axis=1, join="outer")
    if (paired.empty or paired.index.get_level_values("dataset_sha256").nunique() != 1
            or paired.isna().any().any()):
        raise ValueError("Paired contrast requires results from the same dataset and rounds")
    samples = paired.to_numpy() @ np.asarray(list(weights.values()))
    return _estimate(float(samples.mean()), samples)

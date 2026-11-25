"""The single evaluation entry point for the Python baseline."""
import numpy as np
import pandas as pd

from .simulation import expected_profit

METHODS = ("oracle", "naive", "censoring_aware")
METRICS = ("average_expected_profit", "performance_ratio", "regret", "price_mae", "price_bias")


def evaluate_prices(observed_test, prices, config, run):
    pred = observed_test[["sample_id", "x1", "x2", "inventory"]].copy()
    pred["run"] = run
    for method in METHODS:
        price = np.asarray(prices[method])
        if price.shape != (len(pred),):
            raise ValueError(f"Wrong price shape for {method}")
        pred[f"price_{method}"] = price
        pred[f"profit_{method}"] = expected_profit(pred.x1.to_numpy(), pred.x2.to_numpy(),
                                                    pred.inventory.to_numpy(), price, config)
    return pred


def per_run_metrics(pred, run):
    oracle = pred.profit_oracle.mean()
    rows = []
    for method in METHODS:
        profit = pred[f"profit_{method}"].mean()
        difference = pred[f"price_{method}"] - pred.price_oracle
        rows.append({"run": run, "method": method, "average_expected_profit": profit,
                     "performance_ratio": profit/oracle, "regret": oracle-profit,
                     "price_mae": difference.abs().mean(), "price_bias": difference.mean()})
    return rows


def summarize(metrics):
    rows = []
    oracle = metrics.query("method == 'oracle'").sort_values("run").average_expected_profit.to_numpy()
    for method, group in metrics.groupby("method", sort=False):
        group = group.sort_values("run")
        for metric in METRICS:
            values = group[metric].to_numpy()
            if metric == "performance_ratio":
                value = group.average_expected_profit.mean()/oracle.mean()
                influence = (group.average_expected_profit.to_numpy()-value*oracle)/oracle.mean()
                se = influence.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.nan
            else:
                value = values.mean()
                se = values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.nan
            rows.append({"method": method, "metric": metric, "value": value,
                         "standard_error": se, "ci95_low": value-1.96*se,
                         "ci95_high": value+1.96*se, "n_repeats": len(values)})
    return pd.DataFrame(rows)

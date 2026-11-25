from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson, truncnorm

OBSERVED_COLUMNS = ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta"]


def demand_rate(x1, x2, price, config):
    c = config["simulation"]
    return np.maximum(2.5 + np.asarray(x1)**2 + np.asarray(x2)**2
                      - c["price_sensitivity"] * (np.asarray(price) - 0.5), 0.0)



def _inventory(n, mean, std, rng):
    result = np.floor(rng.normal(mean, std, size=n)).astype(int)
    bad = result < 0
    while bad.any():
        result[bad] = np.floor(rng.normal(mean, std, size=bad.sum())).astype(int)
        bad = result < 0
    return result


def generate_dataset(config, seed, n=None):
    c = config["simulation"]
    n = int(n if n is not None else c["n_train"])
    streams = np.random.SeedSequence(seed).spawn(4)
    rx, rp, rd, ry = [np.random.default_rng(s) for s in streams]
    x = rx.uniform(0, 1, size=(n, 2))
    low = (c["price_min"] - c["historical_price_mean"]) / c["historical_price_std"]
    high = (c["price_max"] - c["historical_price_mean"]) / c["historical_price_std"]
    p = truncnorm.rvs(low, high, loc=c["historical_price_mean"],
                      scale=c["historical_price_std"], size=n, random_state=rp)
    rate = demand_rate(x[:, 0], x[:, 1], p, config)
    d = rd.poisson(rate) + 1
    y = _inventory(n, c["inventory_mean"], np.sqrt(c["inventory_variance"]), ry)
    s = np.minimum(d, y)
    delta = (s < y).astype(int)
    observed = pd.DataFrame({"sample_id": np.arange(n), "x1": x[:, 0], "x2": x[:, 1],
                             "price": p, "inventory": y, "sales": s, "delta": delta})
    truth = observed.copy()
    truth["true_demand"] = d
    truth["demand_rate"] = rate
    truth["unmet_demand"] = np.maximum(d-y, 0)
    truth["true_profit"] = p*s-c["stockout_cost"]*truth["unmet_demand"]
    return observed, truth


def expected_profit(x1, x2, inventory, price, config):
    x1, x2, y, p = np.broadcast_arrays(x1, x2, inventory, price)
    lam = demand_rate(x1, x2, p, config)
    sold = np.zeros_like(lam, dtype=float)
    for k in range(1, int(np.max(y)) + 1):
        sold += (y >= k) * poisson.sf(k-2, lam)
    excess = lam + 1 - sold
    return p*sold - config["simulation"]["stockout_cost"]*excess


def data_statistics(truth):
    fields = ["x1", "x2", "price", "inventory", "true_demand", "sales"]
    return {"n": len(truth), "distributions": {
        key: {"mean": float(truth[key].mean()), "std": float(truth[key].std()),
              "min": float(truth[key].min()), "max": float(truth[key].max())}
        for key in fields}, "stockout_rate": float((truth.true_demand >= truth.inventory).mean()),
        "censoring_indicator_rate": float((truth.delta == 0).mean()),
        "tie_rate": float((truth.true_demand == truth.inventory).mean()),
        "unmet_demand_rate": float((truth.unmet_demand > 0).mean()),
        "mean_unmet_demand": float(truth.unmet_demand.mean())}

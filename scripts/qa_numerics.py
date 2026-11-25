"""Independent probability sums for the public pricing evaluation."""

import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy

from pricing_sim.experiment_data import conditional_censored_demand, expected_profit
from pricing_sim.experiment_evaluation import Predictions, evaluate_round, summarize_metrics
from pricing_sim.learning import OraclePricing
from pricing_sim.simulation import expected_profit as e0_expected_profit


def pmf_reference(x1, x2, inventory, price, market):
    """Use explicit PMFs over 512 demand values, with conditional means for possible events."""
    x1, x2, y, p = np.broadcast_arrays(x1, x2, inventory, price)
    rate = 5 + x1*x1 + x2*x2 - 5*p
    k = np.arange(512)
    if market == "poisson":
        log_pmf = xlogy(k, rate[..., None]) - rate[..., None] - gammaln(k + 1)
    elif market == "negative_binomial":
        q = rate[..., None] / (2 + rate[..., None])
        # For k_NB=2, the binomial coefficient is z+1.
        log_pmf = np.log1p(k) + 2*np.log1p(-q) + xlogy(k, q)
    else:
        raise ValueError(market)
    probability = np.exp(log_pmf)
    if not np.allclose(probability.sum(axis=-1), 1, rtol=0, atol=2e-13):
        raise ValueError("Independent PMF sum has insufficient support")
    demand = k + 1
    reward = p[..., None] * np.minimum(demand, y[..., None]) - .1 * np.maximum(demand - y[..., None], 0)
    # Rescale the tail before summing so that very improbable, positive events remain testable.
    log_tail = np.where(demand >= y[..., None], log_pmf, -np.inf)
    log_scale = log_tail.max(axis=-1, keepdims=True)
    scaled = np.full_like(log_tail, -np.inf)
    np.subtract(log_tail, log_scale, out=scaled, where=np.isfinite(log_scale))
    tail = np.exp(scaled)
    denominator = tail.sum(axis=-1)
    mean = np.divide((tail * demand).sum(axis=-1), denominator,
                     out=np.full(rate.shape, np.nan), where=denominator > 0)
    return (probability * reward).sum(axis=-1), mean


def check_release_recovery_targets(data, market):
    """Verify the conditional demand target on every released censored diagnostic row."""
    count = 0
    min_rate, max_error, max_inventory = np.inf, 0., 0
    for rep in range(1, 101):
        diagnostic = data.diagnostics(rep, market).query("delta == 0")
        x1, x2, inventory, price = (diagnostic[c].to_numpy() for c in ("x1", "x2", "inventory", "price"))
        if not len(inventory):
            continue
        _, truth = pmf_reference(x1, x2, inventory, price, market)
        actual = conditional_censored_demand(x1, x2, inventory, price, market)
        np.testing.assert_allclose(actual, truth, rtol=0, atol=2e-10,
                                   err_msg=f"{market} round {rep:03d}")
        count += len(inventory)
        min_rate = min(min_rate, float(np.min(5+x1*x1+x2*x2-5*price)))
        max_error = max(max_error, float(np.max(np.abs(actual-truth))))
        max_inventory = max(max_inventory, int(inventory.max()))
    return {"n_censored": count, "min_lambda": min_rate, "max_inventory": max_inventory,
            "max_abs_error": max_error}


def check_numeric_evaluation(data, e0_data):
    cases = [
        (0., 0., 0, 1.), (0., 0., 1, 1.), (0., 0., 2, 1.),
        (.0001, 0., 3, 1.), (.0001, 0., 60, 1.), (.2, .7, 0, 0.), (.2, .7, 1, 1.),
        (.2, .7, 5, .4), (.8, .6, 10, .7), (1., 1., 35, 0.), (1., 1., 60, 0.),
    ]
    a = np.array(cases).T
    grid = np.linspace(0, 1, 1001)
    max_profit_error, max_mean_error, max_oracle_error = 0., 0., 0.
    tested_edges, impossible_events, grid_rows = 0, 0, 0
    scores, warnings, release = [], [], {}
    for sid, market in (("S05", "poisson"), ("S16", "negative_binomial")):
        reference, reference_mean = pmf_reference(*a, market)
        analytic = expected_profit(*a, market)
        max_profit_error = max(max_profit_error, float(np.max(np.abs(analytic-reference))))
        np.testing.assert_allclose(analytic, reference, rtol=0, atol=1e-11)
        for sample, wanted in zip(cases, reference_mean):
            x1, x2, y, p = sample
            rate = 5 + x1*x1 + x2*x2 - 5*p
            try:
                actual = float(conditional_censored_demand(*sample, market))
            except ValueError as error:
                if rate == 0 and y > 1:
                    impossible_events += 1
                elif 0 < rate < 1e-7 and y == 60 and np.isfinite(wanted):
                    warnings.append({
                        "market": market, "lambda": rate, "inventory": y, "reference": float(wanted),
                        "actual": None, "reason": f"Positive tail underflows in the public evaluator: {error}",
                    })
                else:
                    raise
            else:
                if not np.isfinite(wanted):
                    raise ValueError("An impossible censoring condition must be rejected")
                error = abs(actual-wanted)
                if error <= 2e-10:
                    max_mean_error = max(max_mean_error, float(error))
                elif market == "negative_binomial" and 0 < rate < 1e-7 and y == 3 and error < 1e-6:
                    warnings.append({
                        "market": market, "lambda": rate, "inventory": y, "reference": float(wanted),
                        "actual": actual, "abs_error": float(error),
                        "reason": "Small positive NB rate loses tail precision and yields a mean below inventory",
                    })
                else:
                    np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10, err_msg=str(sample))
        tested_edges += len(cases)
        release[market] = check_release_recovery_targets(data, market)

        cache = {}
        for rep in (1, 100):
            test = data.test(rep)
            diagnostic = data.diagnostics(rep, market)
            state = [test[col].to_numpy() for col in ("x1", "x2", "inventory")]
            hist = [diagnostic[col].to_numpy() for col in ("x1", "x2", "inventory", "price")]
            q, m = pmf_reference(*hist, market)
            censored = diagnostic.delta.eq(0).to_numpy()
            prices = .3 + .3 * state[0]
            pred = Predictions(prices, m[censored] + .25, q + .05)
            evaluated = evaluate_round(data, sid, rep, {"qa_policy": pred}, cache=cache)
            target = cache[(data.manifest_sha256, rep, market)]
            true_profit, _ = pmf_reference(*state, prices, market)
            oracle_profit, _ = pmf_reference(*state, target.oracle_price, market)
            np.testing.assert_allclose(target.oracle_profit, oracle_profit, atol=1e-11, rtol=0)
            for index in (0, 57, 700, 1799):
                x1, x2, y = [value[index] for value in state]
                values, _ = pmf_reference(x1, x2, y, grid, market)
                best = float(values.max())
                price = target.oracle_price[index]
                if price not in grid:
                    raise ValueError("Public Oracle returned a price outside the price grid")
                np.testing.assert_allclose(values[int(round(price*1000))], best, rtol=0, atol=1e-11)
                error = abs(target.oracle_profit[index] - best)
                np.testing.assert_allclose(target.oracle_profit[index], best, rtol=0, atol=1e-11)
                max_oracle_error = max(max_oracle_error, error)
                grid_rows += 1

            for label in ("Overall", "Low", "High"):
                mask = np.ones(len(test), dtype=bool) if label == "Overall" else test.group.eq(label).to_numpy()
                row = evaluated[(evaluated.group == label) & (evaluated.method == "qa_policy")].iloc[0]
                got = [row.profit, row.profit_ratio, row.regret, row.demand_mae, row.q_mae]
                actual, optimum = true_profit[mask].mean(), oracle_profit[mask].mean()
                np.testing.assert_allclose(got, [actual, actual/optimum, optimum-actual, .25, .05],
                                           rtol=0, atol=1e-10)
            scores.append(evaluated)

    metrics = pd.concat(scores, ignore_index=True)
    summary = summarize_metrics(metrics)
    for (sid, label), block in metrics[metrics.method == "qa_policy"].groupby(["scenario_id", "group"]):
        ratio = block.profit.mean() / block.oracle_profit.mean()
        selected = summary.query("scenario_id == @sid and group == @label "
                                 "and method == 'qa_policy' and metric == 'profit_ratio'")
        np.testing.assert_allclose(selected.value.to_numpy(), [ratio], rtol=0, atol=1e-12)

    config = e0_data.manifest["legacy_config"]
    test = e0_data.test(1).iloc[[0, 57, 700, 1799]]
    values = [test[col].to_numpy() for col in ("x1", "x2", "inventory")]
    e0_prices = OraclePricing(config).predict_price(*values)
    e0_profits = e0_expected_profit(*values, e0_prices, config)
    for x1, x2, y, price, profit in zip(*values, e0_prices, e0_profits):
        manual, _ = pmf_reference(x1, x2, y, grid, "poisson")
        if price not in grid:
            raise ValueError("Archived Oracle returned a price outside the price grid")
        np.testing.assert_allclose([manual.max(), manual[int(round(price*1000))]],
                                   [profit, profit], rtol=0, atol=1e-11)
    return {
        "edge_cases": tested_edges, "impossible_conditions": impossible_events,
        "oracle_grid_rows": grid_rows, "e0_oracle_grid_rows": len(test),
        "public_evaluation_calls": len(scores),
        "max_profit_error": max_profit_error, "max_censored_mean_error": max_mean_error,
        "max_oracle_profit_error": max_oracle_error,
        "release_conditional_targets": release, "off_release_warnings": warnings,
    }

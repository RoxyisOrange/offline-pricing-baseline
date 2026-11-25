"""Statistics and standalone figures for the dataset audit."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from scipy.stats import nbinom, norm, poisson

GROUPS = ("Overall", "Low", "High")
EXTREME = ("S05", "S09", "S13", "S16", "S18")
RATE_NAMES = ("overall", "low", "high")


def _describe(values):
    q = np.quantile(values, [.05, .25, .5, .75, .95])
    return {
        "n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        **{f"p{pct:02d}": float(value) for pct, value in zip((5, 25, 50, 75, 95), q)},
    }


def _cdf_distance(values, cdf):
    ordered = np.sort(values)
    n = len(ordered)
    model = cdf(ordered)
    return max(float(np.max(np.arange(1, n + 1) / n - model)),
               float(np.max(model - np.arange(n) / n)))


def distribution_summary(contexts, demand_samples, inventory_rows, calibration):
    rows = []
    price_cdf = lambda x: (ndtr((x - .5) / .5) - ndtr(-1.)) / (ndtr(1.) - ndtr(-1.))
    price_std = .5 * np.sqrt(1 - 2 * norm.pdf(1) / (ndtr(1.) - ndtr(-1.)))
    for split, parts in contexts.items():
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        for name in ("x1", "x2", "price"):
            values = frame[name].to_numpy()
            price = name == "price"
            rows.append({
                "kind": "marginal", "scope": split, "field": name,
                **_describe(values), "theory_mean": .5, "theory_std": price_std if price else np.sqrt(1/12),
                "cdf_max_distance": _cdf_distance(values, price_cdf if price else lambda x: x),
                "at_zero": int(np.sum(values == 0)), "at_one": int(np.sum(values == 1)),
            })
        for label in GROUPS:
            group = frame if label == "Overall" else frame[frame.group == label]
            rows.append({"kind": "feature_correlation", "scope": split, "group": label,
                         "field": "x1,x2", "n": len(group), "correlation": group.x1.corr(group.x2)})

    for market, parts in demand_samples.items():
        if not parts:
            continue
        data = pd.concat(parts, ignore_index=True)
        bins = pd.cut(data["lambda"], [0, 1, 2, 3, 4, 5, 6, 7], include_lowest=True)
        blocks = [("all", data), *((str(k), v) for k, v in data.groupby(bins, observed=True))]
        for label, subset in blocks:
            rate = subset["lambda"].to_numpy()
            demand = subset.demand.to_numpy()
            mean = 1 + rate
            variance = rate if market == "poisson" else rate + rate*rate/2
            residual = demand - mean
            rows.append({
                "kind": "conditional_demand", "scope": market, "group": label, "field": "demand",
                **_describe(demand), "theory_mean": mean.mean(), "mean_residual": residual.mean(),
                "mean_conditional_variance": variance.mean(), "mean_squared_residual": np.mean(residual**2),
                "second_moment_ratio": np.mean(residual**2) / variance.mean(),
            })
    rows.extend(inventory_rows)
    rows.extend(calibration)
    return pd.DataFrame(rows)


def check_calibration(table):
    """Integrate P(D>=Y) over X and P by Gauss–Legendre quadrature, independently of the Sobol fit."""
    nodes, weights = leggauss(28)
    u, w = (nodes + 1) / 2, weights / 2
    angle = u * np.pi / 4
    price_weights = w * norm.pdf(u, loc=.5, scale=.5) / (ndtr(1.) - ndtr(-1.))
    threshold = 2 / np.pi
    rows = []
    for label in ("Low", "High"):
        z_min = 0 if label == "Low" else threshold
        z_max = np.full_like(angle, threshold) if label == "Low" else 1 / np.cos(angle)**2
        width = z_max - z_min
        z = z_min + width[:, None] * u[None, :]
        # Reflection about pi/4 doubles the area element (1/2) dz dtheta.
        area = (w * np.pi / 4)[:, None] * width[:, None] * w[None, :]
        rates = (5 + z[:, :, None] - 5*u).ravel()
        joint_weight = (2 * area[:, :, None] * price_weights).ravel()
        for market in ("poisson", "negative_binomial"):
            survival = np.array([
                joint_weight @ (poisson.sf(y - 2, rates) if market == "poisson"
                                else nbinom.sf(y - 2, 2, 2/(2 + rates)))
                for y in range(65)
            ])
            selection = table[(table.group == label) & (table.market == market)]
            for row in selection.itertuples():
                mu = row.inventory_mean_parameter
                lower = ndtr(-mu / np.sqrt(2))
                cdf = (ndtr((np.arange(66) - mu) / np.sqrt(2)) - lower) / (1 - lower)
                achieved = float(survival @ np.diff(cdf))
                rows.append({"kind": "inventory_calibration", "scope": market, "group": label,
                             "field": "Pr(D>=Y)", "target_rate": row.target_rate,
                             "inventory_mean_parameter": mu, "integral_rate": achieved,
                             "deviation_pp": 100 * (achieved - row.target_rate)})
    return rows


class Coverage:
    def __init__(self):
        self.histograms = defaultdict(Counter)
        self.rounds = []

    def add(self, sid, rep, observed, demand, test):
        all_max = int(observed.inventory.max())
        fields = {}
        for label in GROUPS:
            mask = np.ones(len(observed), dtype=bool) if label == "Overall" else observed.group.eq(label).to_numpy()
            train = observed.loc[mask]
            test_group = test if label == "Overall" else test[test.group == label]
            y, y_test = train.inventory.to_numpy(), test_group.inventory.to_numpy()
            d = demand[mask]
            stockout = train.delta.eq(0).to_numpy()
            n_censored = int(stockout.sum())
            no_info = int(np.sum(stockout & (y <= 1)))
            loss = np.maximum(d - y, 0)
            top = int(y.max())
            row = {
                "scenario_id": sid, "round_id": rep, "group": label,
                "n_train": len(train), "n_test": len(test_group), "n_censored": n_censored,
                "n_complete": len(train) - n_censored, "zero_inventory_rate": np.mean(y == 0),
                "zero_sales_rate": train.sales.eq(0).mean(), "inventory_at_most_one_rate": np.mean(y <= 1),
                "complete_rate": np.mean(~stockout), "no_extra_info_rate": no_info / len(train),
                "no_extra_info_among_censored": no_info / n_censored if n_censored else np.nan,
                "tie_rate": np.mean(d == y), "unmet_units": int(loss.sum()),
                "unmet_per_record": loss.mean(),
                "mean_unmet_given_censored": loss[stockout].mean() if n_censored else np.nan,
                "train_inventory_min": int(y.min()), "train_inventory_max": top,
                "test_inventory_min": int(y_test.min()), "test_inventory_max": int(y_test.max()),
                "test_above_pooled_train_max_rate": np.mean(y_test > all_max),
                "test_above_group_train_max_rate": np.mean(y_test > top),
            }
            self.rounds.append(row)
            for name, values in (("train", y), ("test", y_test)):
                hist = np.bincount(values)
                self.histograms[(sid, label, name)].update({k: int(n) for k, n in enumerate(hist) if n})
            fields.update({f"{key}_{label.lower()}": value for key, value in row.items()
                           if key not in ("scenario_id", "round_id", "group")})
        return fields

    def summary(self):
        rounds = pd.DataFrame(self.rounds)
        summary = []
        marginal = []
        if rounds.empty:
            return pd.DataFrame(), []
        for (sid, label), block in rounds.groupby(["scenario_id", "group"], sort=False):
            row = {"scenario_id": sid, "group": label, "rounds": len(block)}
            numeric = block.drop(columns=["scenario_id", "round_id", "group"])
            for key in numeric.columns:
                column = numeric[key]
                row[key] = (int(column.sum()) if key in {"n_train", "n_test", "n_censored", "n_complete", "unmet_units"}
                            else int(column.min()) if key.endswith("_min")
                            else int(column.max()) if key.endswith("_max")
                            else float(column.mean()))
            for name in ("train", "test"):
                histogram = self.histograms[(sid, label, name)]
                values = np.repeat(list(histogram), list(histogram.values()))
                stats = _describe(values)
                row.update({f"{name}_inventory_{k}": v for k, v in stats.items() if k != "n"})
                marginal.append({"kind": "inventory", "scope": sid, "group": label, "field": name, **stats})
            train = self.histograms[(sid, label, "train")]
            test = self.histograms[(sid, label, "test")]
            row["inventory_total_variation"] = .5 * sum(
                abs(train[k] / train.total() - test[k] / test.total()) for k in train.keys() | test.keys()
            )
            summary.append(row)
        return pd.DataFrame(summary), marginal


def summarize_censoring(per_round):
    if per_round.empty:
        return pd.DataFrame()
    rows = []
    for (sid, market), block in per_round.groupby(["scenario_id", "market"], sort=False):
        row = {"scenario_id": sid, "market": market, "n_rounds": len(block)}
        for label in RATE_NAMES:
            values = block[f"actual_{label}_rate"]
            target = block[f"target_{label}_rate"].iloc[0]
            row.update({
                f"target_{label}_rate": target, f"actual_{label}_mean": values.mean(),
                f"actual_{label}_std": values.std(ddof=1), f"actual_{label}_min": values.min(),
                f"actual_{label}_max": values.max(), f"{label}_deviation_pp": (values.mean() - target) * 100,
                f"n_censored_{label}": int(block[f"n_censored_{label}"].sum()),
                f"n_complete_{label}": int(block[f"n_complete_{label}"].sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def make_figures(output: Path, per_round, coverage, coverage_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir()
    saved = []
    if not per_round.empty:
        fig, axes = plt.subplots(6, 3, figsize=(12, 13), sharex=True)
        for ax, (sid, block) in zip(axes.flat, per_round.groupby("scenario_id", sort=True)):
            ax.plot(block.round_id, block.actual_low_rate * 100, color="#225ea8", lw=.75, label="Low")
            ax.plot(block.round_id, block.actual_high_rate * 100, color="#d15c29", lw=.75, label="High")
            for label, color in (("low", "#225ea8"), ("high", "#d15c29")):
                ax.axhline(block[f"target_{label}_rate"].iloc[0] * 100, color=color, ls=":", lw=1.1)
            ax.set_title(sid, loc="left", fontweight="bold", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Observed censoring by round and target", fontsize=17, fontweight="bold")
        fig.supxlabel("Replication")
        fig.supylabel("Censored rows (%)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .02, 1, .97))
        saved.append("censoring_by_round.png")
        fig.savefig(output / saved[-1], dpi=160)
        plt.close(fig)

    if not coverage_summary.empty:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
        for ax, sid in zip(axes.flat, EXTREME):
            for name, color in (("train", "#225ea8"), ("test", "#d15c29")):
                hist = coverage.histograms[(sid, "Overall", name)]
                if hist:
                    support = np.arange(max(hist) + 1)
                    ax.step(support, [hist[i] / hist.total() * 100 for i in support], where="mid",
                            lw=1.7, color=color, label=name.title())
            ax.set_title(sid, loc="left", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
        axes.flat[-1].axis("off")
        axes.flat[0].legend(frameon=False)
        fig.suptitle("Inventory distributions in high-censoring scenarios", fontsize=16, fontweight="bold")
        fig.supxlabel("Inventory units")
        fig.supylabel("Share of records (%)")
        fig.tight_layout(rect=(0, .03, 1, .96))
        saved.append("inventory_distribution.png")
        fig.savefig(output / saved[-1], dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        labels = sorted(coverage_summary.scenario_id.unique())
        width = .25
        positions = np.arange(len(labels))
        for offset, label, color in ((-width, "Overall", "#6b7683"), (0, "Low", "#225ea8"),
                                      (width, "High", "#d15c29")):
            selected = coverage_summary[coverage_summary.group == label].set_index("scenario_id")
            share = selected["test_above_group_train_max_rate"].reindex(labels) * 100
            ax.bar(positions + offset, share, width, label=label, color=color)
        ax.set_xticks(positions, labels, fontsize=9)
        ax.set_ylabel("Test rows above training maximum (%)")
        ax.set_title("Inventory coverage by scenario and group", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        saved.append("inventory_coverage.png")
        fig.savefig(output / saved[-1], dpi=170)
        plt.close(fig)
    return saved

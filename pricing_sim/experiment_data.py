"""Versioned, paired data for the E1–E4 experiment guide."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.stats import nbinom, poisson, qmc, truncnorm

from .data_store import sha256
from .simulation import OBSERVED_COLUMNS

VERSION = "2.0.0"
N_TRAIN = N_TEST = 2000
N_ROUNDS = 100
MASTER_SEED, PILOT_SEED = 20261125, 20261126
PILOT_LOG2_N = 17
PRICE_MEAN, PRICE_SD = 0.5, 0.5
INVENTORY_MEAN, INVENTORY_VARIANCE = 6.0, 2.0
INVENTORY_SD = np.sqrt(INVENTORY_VARIANCE)
STOCKOUT_COST = 0.1
NB_SIZE = 2.0
ORACLE_STEP = 0.001
GROUP_THRESHOLD = 2 / np.pi
GROUPS = ("Low", "High")
MARKETS = ("poisson", "negative_binomial")
TRAIN_COLUMNS = (*OBSERVED_COLUMNS, "group")
TEST_COLUMNS = ("sample_id", "x1", "x2", "inventory", "group")
TRUTH_COLUMNS = ("sample_id", "true_demand", "demand_rate", "mean_demand", "true_profit")
_DESIGN = {
    "master_seed": MASTER_SEED, "pilot_seed": PILOT_SEED, "pilot_log2_n": PILOT_LOG2_N,
    "n_train": N_TRAIN, "n_test": N_TEST, "group_threshold": GROUP_THRESHOLD,
    "n_per_train_group": N_TRAIN // 2, "n_per_test_group": N_TEST // 2,
    "inventory_mean_test": INVENTORY_MEAN, "inventory_variance": INVENTORY_VARIANCE,
    "price_mean": PRICE_MEAN, "price_sd": PRICE_SD, "nb_size": NB_SIZE,
    "stockout_cost": STOCKOUT_COST, "oracle_step": ORACLE_STEP,
}


@dataclass(frozen=True)
class Scenario:
    market: str
    low_target_pct: int
    high_target_pct: int

    @property
    def directory(self) -> Path:
        return Path(self.market) / f"{self.low_target_pct}_{self.high_target_pct}"


_POISSON_PAIRS = ((10, 10), (30, 30), (50, 50), (70, 70), (90, 90),
                  (40, 60), (30, 70), (20, 80), (10, 90),
                  (20, 40), (10, 50), (60, 80), (50, 90))
_NB_PAIRS = ((10, 10), (50, 50), (90, 90), (30, 70), (10, 90))
SCENARIOS = MappingProxyType({
    f"S{i:02d}": spec for i, spec in enumerate(
        [Scenario("poisson", *pair) for pair in _POISSON_PAIRS]
        + [Scenario("negative_binomial", *pair) for pair in _NB_PAIRS], start=1
    )
})
EXPERIMENTS = MappingProxyType({
    "E1": ("S01", "S02", "S03", "S04", "S05"),
    "E2": ("S03", "S06", "S07", "S08", "S09"),
    "E3": ("S02", "S10", "S11", "S03", "S06", "S07", "S04", "S12", "S13"),
    "E4.1": ("S01", "S03", "S05", "S07", "S09"),
    "E4.2": ("S14", "S15", "S16", "S17", "S18"),
})


@dataclass(frozen=True)
class TrainingSample:
    observed: pd.DataFrame
    truth: pd.DataFrame
    statistics: dict


def _check_round(round_id: int, limit: int = N_ROUNDS) -> int:
    if (isinstance(round_id, bool) or not isinstance(round_id, (int, np.integer))
            or not 1 <= round_id <= limit):
        raise ValueError(f"Round must be an integer in [1, {limit}]")
    return int(round_id)


def _rng(round_id, split, stream):
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, round_id, split, stream]))


def _low_group(x):
    return x[:, 0] ** 2 + x[:, 1] ** 2 <= GROUP_THRESHOLD


def _balanced_features(n, rng):
    # Sample from the original uniform distribution conditional on each fixed group.
    half = n // 2
    x = np.empty((n, 2))
    counts = [0, 0]
    while min(counts) < half:
        candidates = rng.random((n, 2))
        low = _low_group(candidates)
        for group, mask in enumerate((low, ~low)):
            accepted = candidates[mask][:half - counts[group]]
            start = group * half + counts[group]
            x[start:start + len(accepted)] = accepted
            counts[group] += len(accepted)
    return rng.permutation(x, axis=0)


def _price(u):
    return truncnorm.ppf(u, -PRICE_MEAN / PRICE_SD, (1 - PRICE_MEAN) / PRICE_SD,
                         loc=PRICE_MEAN, scale=PRICE_SD)


def _inventory(u, mean):
    # A nonnegative continuous draw is floored. Zero inventory is retained.
    return np.floor(truncnorm.ppf(u, -mean / INVENTORY_SD, np.inf,
                                 loc=mean, scale=INVENTORY_SD)).astype(np.int64)


def _base(round_id, split):
    n = N_TRAIN if split == 0 else N_TEST
    x = _balanced_features(n, _rng(round_id, split, 10))
    price = _price(_rng(round_id, split, 20).random(n))
    group = np.where(_low_group(x), *GROUPS)
    return x, price, group, _rng(round_id, split, 40).random(n)


def demand_rate(x1, x2, price):
    x1, x2, price = np.broadcast_arrays(np.asarray(x1, float),
                                       np.asarray(x2, float), np.asarray(price, float))
    if any(not np.all((a >= 0) & (a <= 1)) for a in (x1, x2, price)):
        raise ValueError("Features and price must be in [0,1]")
    return 5 + x1**2 + x2**2 - 5 * price


def _draw_demand(rate, round_id, split, market):
    if market == "poisson":
        return 1 + _rng(round_id, split, 30).poisson(rate)
    if market == "negative_binomial":
        return 1 + _rng(round_id, split, 31).negative_binomial(NB_SIZE, NB_SIZE / (NB_SIZE + rate))
    raise ValueError(f"Unknown market {market!r}")


def _tail(cutoff, rate, market, size_offset=0):
    """P(Z > cutoff) for D=1+Z, with the NB first-moment size shift if requested."""
    if market == "poisson":
        return poisson.sf(cutoff, rate)
    if market == "negative_binomial":
        return nbinom.sf(cutoff, NB_SIZE + size_offset, NB_SIZE / (NB_SIZE + rate))
    raise ValueError(f"Unknown market {market!r}")


@lru_cache(maxsize=None)
def _pilot_tails(market, group):
    # Integrate over independent X and P from a fixed pilot. No realized demand enters calibration.
    u = qmc.Sobol(d=3, scramble=True, seed=PILOT_SEED).random_base2(PILOT_LOG2_N)
    low = _low_group(u[:, :2])
    rows = u[low if group == "Low" else ~low]
    rates = demand_rate(rows[:, 0], rows[:, 1], _price(rows[:, 2]))
    # P(D >= Y) = P(Z > Y-2). The cutoff is only for the numerical integration of Y.
    return np.array([_tail(y - 2, rates, market).mean() for y in range(65)])


@lru_cache(maxsize=None)
def _calibrate(market, group, target_pct):
    tails = _pilot_tails(market, group)
    edges = np.arange(len(tails) + 1)

    def rate_at(mean):
        cdf = truncnorm.cdf(edges, -mean / INVENTORY_SD, np.inf,
                            loc=mean, scale=INVENTORY_SD)
        if 1 - cdf[-1] > 1e-12:
            raise ValueError("Integration range too short for inventory calibration")
        return np.diff(cdf) @ tails

    return float(brentq(lambda mean: rate_at(mean) - target_pct / 100,
                        -12.0, 32.0, xtol=1e-10))


def _observed(x, price, group, inventory, demand):
    sales = np.minimum(demand, inventory)
    return pd.DataFrame({
        "sample_id": np.arange(1, len(x) + 1),
        "x1": x[:, 0], "x2": x[:, 1], "price": price, "inventory": inventory,
        "sales": sales, "delta": (sales < inventory).astype(np.int8), "group": group,
    }, columns=TRAIN_COLUMNS)


def generate_train(scenario_id: str, round_id: int) -> TrainingSample:
    """Generate a paired training sample. Ordinary fitting uses observed fields only."""
    round_id = _check_round(round_id)
    scenario = SCENARIOS[scenario_id]
    x, price, group, inventory_u = _base(round_id, 0)
    rate = demand_rate(x[:, 0], x[:, 1], price)
    demand = _draw_demand(rate, round_id, 0, scenario.market)
    mean_low = _calibrate(scenario.market, "Low", scenario.low_target_pct)
    mean_high = _calibrate(scenario.market, "High", scenario.high_target_pct)
    inventory = _inventory(inventory_u, np.where(group == "Low", mean_low, mean_high))
    observed = _observed(x, price, group, inventory, demand)
    truth = pd.DataFrame({
        "sample_id": observed.sample_id, "true_demand": demand, "demand_rate": rate,
        "mean_demand": 1 + rate,
        "true_profit": price * observed.sales - STOCKOUT_COST * np.maximum(demand - inventory, 0),
    }, columns=TRUTH_COLUMNS)
    censored = observed.delta.to_numpy() == 0
    statistics = {
        "scenario_id": scenario_id, "round_id": round_id, "market": scenario.market,
        "r_low_target": scenario.low_target_pct / 100, "r_high_target": scenario.high_target_pct / 100,
        "r_low_actual": float(censored[group == "Low"].mean()),
        "r_high_actual": float(censored[group == "High"].mean()),
        "r_actual": float(censored.mean()), "zero_inventory_rate": float(np.mean(inventory == 0)),
        "inventory_mean_parameter_low": mean_low, "inventory_mean_parameter_high": mean_high,
    }
    return TrainingSample(observed, truth, statistics)


def _test(round_id, market=None):
    round_id = _check_round(round_id)
    x, price, group, u = _base(round_id, 1)
    inventory = _inventory(u, INVENTORY_MEAN)
    contexts = pd.DataFrame({
        "sample_id": np.arange(1, len(x) + 1),
        "x1": x[:, 0], "x2": x[:, 1], "inventory": inventory, "group": group,
    }, columns=TEST_COLUMNS)
    if market is None:
        return contexts
    rate = demand_rate(x[:, 0], x[:, 1], price)
    demand = _draw_demand(rate, round_id, 1, market)
    return _observed(x, price, group, inventory, demand)


def generate_test(round_id: int) -> pd.DataFrame:
    """Shared X and inventory used by every scenario and market in the round."""
    return _test(round_id)


def generate_diagnostic_test(round_id: int, market: str) -> pd.DataFrame:
    """Held-out logged prices and sales for demand and Q evaluation."""
    return _test(round_id, market)


def _profit_inputs(x1, x2, inventory, price):
    rate, inventory, price = np.broadcast_arrays(
        demand_rate(x1, x2, price), np.asarray(inventory, float), np.asarray(price, float))
    if not np.all(np.isfinite(inventory) & (inventory >= 0) & (inventory == np.floor(inventory))):
        raise ValueError("Inventory must be a nonnegative integer")
    return rate, inventory, price


def expected_profit(x1, x2, inventory, price, market):
    """Evaluate E[p min(D,Y) - c max(D-Y,0)] at the supplied price."""
    rate, inventory, price = _profit_inputs(x1, x2, inventory, price)
    cutoff = inventory - 1
    tail = _tail(cutoff, rate, market)
    first_moment = rate * _tail(cutoff - 1, rate, market, size_offset=1)
    excess = np.maximum(first_moment - cutoff * tail, 0)
    sold = np.clip(1 + rate - excess, 0, inventory)
    return price * sold - STOCKOUT_COST * excess


def conditional_censored_demand(x1, x2, inventory, price, market):
    """Evaluate the true E[D | X,P,D >= Y] on held-out sold-out records."""
    rate, inventory, _ = _profit_inputs(x1, x2, inventory, price)
    cutoff = inventory - 2
    tail = _tail(cutoff, rate, market)
    if np.any(tail == 0):
        raise ValueError("The censored demand event has zero probability")
    return 1 + rate * _tail(cutoff - 1, rate, market, size_offset=1) / tail


def self_test(rounds=N_ROUNDS):
    """Check balances, pairing and achieved censoring rates across the suite."""
    rounds = _check_round(rounds)
    stats = []
    pairing_columns = ["sample_id", "x1", "x2", "price", "group"]
    for round_id in range(1, rounds + 1):
        test = generate_test(round_id)
        if test.group.value_counts().to_dict() != dict.fromkeys(GROUPS, N_TEST // 2):
            raise ValueError(f"Unbalanced test in round {round_id}")
        for market in MARKETS:
            diag = generate_diagnostic_test(round_id, market)
            pd.testing.assert_frame_equal(test, diag[list(TEST_COLUMNS)])
        contexts, demand_by_market = None, {}
        for sid, scenario in SCENARIOS.items():
            sample = generate_train(sid, round_id)
            observed, truth = sample.observed, sample.truth
            if observed.group.value_counts().to_dict() != dict.fromkeys(GROUPS, N_TRAIN // 2):
                raise ValueError(f"Unbalanced training data: {sid}, round {round_id}")
            if (observed[["x1", "x2"]].equals(test[["x1", "x2"]])
                    or not np.array_equal(observed.sales, np.minimum(truth.true_demand, observed.inventory))
                    or not np.array_equal(observed.delta, (truth.true_demand < observed.inventory).astype(int))
                    or not np.all(observed.inventory >= 0)):
                raise ValueError(f"Invalid sales, censoring or split: {sid}, round {round_id}")
            if contexts is None:
                contexts = observed[pairing_columns]
            pd.testing.assert_frame_equal(contexts, observed[pairing_columns])
            reference = demand_by_market.setdefault(scenario.market, truth.true_demand)
            pd.testing.assert_series_equal(reference, truth.true_demand)
            stats.append(sample.statistics)
    report = pd.DataFrame(stats).groupby(["scenario_id", "market"])[[
        "r_low_target", "r_high_target", "r_low_actual", "r_high_actual",
        "r_actual", "zero_inventory_rate",
    ]].mean()
    tolerance = max(0.006, 4 * np.sqrt(0.25 / (rounds * (N_TRAIN // 2))))
    for group in ("low", "high"):
        if ((report[f"r_{group}_actual"] - report[f"r_{group}_target"]).abs() > tolerance).any():
            raise ValueError(f"Mean {group} group censoring outside sampling tolerance")
    print(report.to_string(float_format=lambda value: f"{value:.5f}"))
    print(f"PASS: {len(stats)} training sets and {rounds} shared test sets")
    return report


def export_data(output: Path, rounds: int, scenarios=tuple(SCENARIOS)) -> dict:
    """Freeze observations, shared tests and separately stored evaluation data."""
    rounds = _check_round(rounds)
    scenarios = tuple(dict.fromkeys(scenarios))
    if not scenarios or set(scenarios).difference(SCENARIOS):
        raise ValueError("Choose at least one registered scenario")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files, statistics = {}, []

    def write(relative: Path, frame: pd.DataFrame):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.17g")
        files[relative.as_posix()] = sha256(path)

    markets = sorted({SCENARIOS[sid].market for sid in scenarios})
    for round_id in range(1, rounds + 1):
        write(Path("shared_test") / f"test_{round_id:03d}.csv", generate_test(round_id))
        for market in markets:
            write(Path("evaluation_only", market, f"diagnostic_{round_id:03d}.csv"),
                  generate_diagnostic_test(round_id, market))
        for sid in scenarios:
            sample = generate_train(sid, round_id)
            directory = SCENARIOS[sid].directory
            write(directory / f"train_{round_id:03d}.csv", sample.observed)
            write(Path("evaluation_only") / directory / f"truth_{round_id:03d}.csv", sample.truth)
            statistics.append(sample.statistics)
    write(Path("actual_censoring_rates.csv"), pd.DataFrame(statistics))

    manifest = {
        "version": VERSION, "generator_sha256": sha256(Path(__file__)), "rounds": rounds,
        **_DESIGN,
        "scenarios": {
            sid: {"market": SCENARIOS[sid].market, "directory": SCENARIOS[sid].directory.as_posix(),
                  "low_target_pct": SCENARIOS[sid].low_target_pct,
                  "high_target_pct": SCENARIOS[sid].high_target_pct}
            for sid in scenarios
        },
        "experiments": dict(EXPERIMENTS),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "files": files,
    }
    # A manifest only becomes visible after the entire export is complete.
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest


class ExperimentData:
    """Check the pinned data version and load just the tables a caller needs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        raw = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        if self.manifest["version"] != VERSION:
            raise ValueError(f"Expected experiment data generator {VERSION}")
        for key, value in _DESIGN.items():
            if self.manifest.get(key) != value:
                raise ValueError(f"Frozen experiment parameter differs: {key}")
        self.rounds = _check_round(self.manifest["rounds"])
        self.scenarios = tuple(self.manifest["scenarios"])
        for sid, row in self.manifest["scenarios"].items():
            spec = SCENARIOS[sid]
            if (row["market"], row["low_target_pct"], row["high_target_pct"], row["directory"]) != (
                    spec.market, spec.low_target_pct, spec.high_target_pct, spec.directory.as_posix()):
                raise ValueError(f"Scenario settings differ for {sid}")
        self._statistics = None

    def _read(self, relative: Path, columns=None, size=None):
        path = self.root / relative
        if sha256(path) != self.manifest["files"].get(relative.as_posix()):
            raise ValueError(f"Frozen file checksum changed: {relative}")
        frame = pd.read_csv(path, float_precision="round_trip")
        if columns is not None and (tuple(frame.columns) != columns or len(frame) != size):
            raise ValueError(f"Unexpected frozen table schema: {relative}")
        return frame

    def training(self, scenario_id: str, round_id: int) -> pd.DataFrame:
        """Return the exact E0 training schema, without group labels or true quantities."""
        round_id = _check_round(round_id, self.rounds)
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} absent from export")
        path = SCENARIOS[scenario_id].directory / f"train_{round_id:03d}.csv"
        return self._read(path, TRAIN_COLUMNS, N_TRAIN)[OBSERVED_COLUMNS]

    def test(self, round_id: int) -> pd.DataFrame:
        round_id = _check_round(round_id, self.rounds)
        return self._read(Path("shared_test", f"test_{round_id:03d}.csv"), TEST_COLUMNS, N_TEST)

    def diagnostics(self, round_id: int, market: str) -> pd.DataFrame:
        round_id = _check_round(round_id, self.rounds)
        if market not in MARKETS:
            raise ValueError(f"Unknown market {market!r}")
        path = Path("evaluation_only", market, f"diagnostic_{round_id:03d}.csv")
        return self._read(path, TRAIN_COLUMNS, N_TEST)

    def rates(self, scenario_id: str, round_id: int) -> dict:
        if self._statistics is None:
            self._statistics = self._read(Path("actual_censoring_rates.csv")).set_index(
                ["scenario_id", "round_id"], verify_integrity=True)
        row = self._statistics.loc[(scenario_id, round_id)]
        return {key: float(row[key]) for key in ("r_low_actual", "r_high_actual", "r_actual")}

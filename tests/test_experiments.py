import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from pricing_sim.experiment_data import (GROUP_THRESHOLD, ExperimentData,
                                         conditional_censored_demand, expected_profit,
                                         export_data, generate_test, generate_train)
from pricing_sim.experiment_evaluation import (Predictions, evaluate_fitted_models,
                                               evaluate_round, paired_contrast, summarize_metrics)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments") / "rounds"
    (root / "v1").mkdir(parents=True)
    (root / "v1" / "kept.txt").write_text("original E0")
    export_data(root, rounds=1, scenarios=("S01", "S05", "S16"))
    return ExperimentData(root)


def test_balanced_split_pairs_the_same_rows_across_scenarios():
    s01, s05, s16 = [generate_train(sid, 1) for sid in ("S01", "S05", "S16")]
    paired = ["sample_id", "x1", "x2", "price", "group"]
    for sample in (s01, s05, s16):
        obs, truth = sample.observed, sample.truth
        assert obs.group.value_counts().to_dict() == {"Low": 1000, "High": 1000}
        np.testing.assert_array_equal(obs.group == "Low", obs.x1**2 + obs.x2**2 <= GROUP_THRESHOLD)
        np.testing.assert_array_equal(obs.sales, np.minimum(truth.true_demand, obs.inventory))
        np.testing.assert_array_equal(obs.delta == 0, truth.true_demand >= obs.inventory)
        pd.testing.assert_frame_equal(s01.observed[paired], obs[paired])

    pd.testing.assert_series_equal(s01.truth.true_demand, s05.truth.true_demand)
    assert (s01.observed.inventory >= s05.observed.inventory).all()
    assert (s16.observed.inventory == 0).any()
    assert (s16.observed.loc[s16.observed.inventory == 0, "delta"] == 0).all()
    test = generate_test(1)
    assert test.group.value_counts().to_dict() == {"Low": 1000, "High": 1000}
    assert not test[["x1", "x2"]].equals(s01.observed[["x1", "x2"]])
    assert not test.equals(generate_test(2))


@pytest.mark.parametrize("market", ["poisson", "negative_binomial"])
def test_profit_and_conditional_demand_match_distribution_enumeration(market):
    x1, x2, price = .2, .7, .6
    rate = 5 + x1**2 + x2**2 - 5 * price
    demand = 1 + np.arange(1000)
    distribution = poisson(rate) if market == "poisson" else nbinom(2, 2 / (2 + rate))
    probability = distribution.pmf(demand - 1)
    for y in (0, 1, 2, 5, 11):
        true_profit = probability @ (price * np.minimum(demand, y) - .1 * np.maximum(demand-y, 0))
        tail = demand >= y
        true_mean = (probability[tail] @ demand[tail]) / probability[tail].sum()
        assert expected_profit(x1, x2, y, price, market) == pytest.approx(true_profit, abs=1e-12)
        assert conditional_censored_demand(x1, x2, y, price, market) == pytest.approx(true_mean, abs=1e-12)


def test_export_is_write_once_and_loader_requires_no_truth(data):
    truth = data.root / "ground_truth/negative_binomial/S16_L90_H90/truth_001.csv"
    withheld = truth.with_suffix(".withheld")
    truth.rename(withheld)
    try:
        train = data.training("S16", 1)
    finally:
        withheld.rename(truth)
    assert list(train) == ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta"]
    assert len(train) == 2000
    with pytest.raises(FileExistsError):
        export_data(data.root, rounds=1, scenarios=("S16",))

    path = data.root / "train/negative_binomial/S16_L90_H90/train_001.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="checksum"):
            data.training("S16", 1)
    finally:
        path.write_bytes(original)


def test_release_preserves_legacy_data_and_records_quality(data):
    from data_loader import get_scenarios, load_test, load_train

    assert (data.root / "v1" / "kept.txt").read_text() == "original E0"
    assert get_scenarios("E2") == ["S03", "S06", "S07", "S08", "S09"]
    assert get_scenarios("E0") == ["E0"]
    pd.testing.assert_frame_equal(load_train("S16", 1, data_dir=data.root), data.training("S16", 1))
    pd.testing.assert_frame_equal(load_test(1, data_dir=data.root), data.test(1))
    assert (data.root / "metadata" / "checksums.sha256").is_file()
    quality = pd.read_csv(data.root / "metadata" / "data_quality_report.csv")
    assert quality.n_low.eq(1000).all() and quality.n_high.eq(1000).all()
    assert quality.missing_values.eq(0).all()
    assert set(quality["round"]) == {1}


def test_evaluation_uses_shared_context_but_market_specific_truth(data):
    cache = {}
    test = data.test(1)
    evals = []
    for sid, market in (("S01", "poisson"), ("S05", "poisson"), ("S16", "negative_binomial")):
        diag = data.diagnostics(1, market)
        censored = diag[diag.delta == 0]
        predictions = {
            "naive": Predictions(np.full(len(test), 0.5)),
            "censored_poisson": Predictions(
                np.full(len(test), 0.6),
                conditional_censored_demand(censored.x1, censored.x2, censored.inventory,
                                            censored.price, market),
                expected_profit(diag.x1, diag.x2, diag.inventory, diag.price, market),
            ),
        }
        result = evaluate_round(data, sid, 1, predictions, cache=cache)
        perfect = result[result.method == "censored_poisson"]
        assert len(perfect) == 3
        np.testing.assert_allclose(perfect[["demand_mae", "q_mae"]], 0)
        naive = result.query("method == 'naive' and group == 'Overall'").iloc[0]
        assert naive.profit == pytest.approx(
            expected_profit(test.x1, test.x2, test.inventory, .5, market).mean())
        assert naive.demand_mae > 0
        evals.append(result)
    assert len(cache) == 2
    oracle = [frame.query("method == 'oracle' and group == 'Overall'").profit.iloc[0] for frame in evals]
    assert oracle[0] == oracle[1]
    assert oracle[0] != oracle[2]


def test_fitted_model_evaluation_reads_only_held_out_observations(data):
    class FittedPolicy:
        def predict_price(self, x1, x2, inventory):
            return np.full(len(x1), .5)

        def predict_censored_demand(self, diagnostic):
            return diagnostic.loc[diagnostic.delta == 0, "sales"].to_numpy()

        def predict_q(self, diagnostic):
            assert not {"true_demand", "group"} & set(diagnostic)
            return diagnostic.price.to_numpy() * diagnostic.sales.to_numpy()

    score = evaluate_fitted_models(data, "S16", 1, {"naive": FittedPolicy()})
    assert score.loc[score.method == "naive", "q_mae"].notna().all()


def test_aggregate_ratios_and_contrasts_pair_rounds():
    rows = []
    for round_id, oracle, naive, aware in ((1, 1., 0.5, 0.8), (2, 3., 2.7, 2.5)):
        for method, profit in (("naive", naive), ("censoring_aware", aware)):
            rows.append({
                "scenario_id": "S01", "round_id": round_id, "market": "poisson",
                "generator_version": "2.0.0", "dataset_sha256": "example", "group": "Overall",
                "method": method, "profit": profit, "oracle_profit": oracle,
                "profit_ratio": profit/oracle, "regret": oracle-profit,
                "correction_value": profit-naive, "price_mae": np.nan, "price_bias": np.nan,
                "demand_mae": np.nan, "q_mae": np.nan,
            })
    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)
    naive_ratio = summary.query("method == 'naive' and metric == 'profit_ratio'").value.iloc[0]
    assert naive_ratio == pytest.approx(.8)
    contrast = paired_contrast(metrics, {("S01", "censoring_aware"): 1, ("S01", "naive"): -1})
    assert contrast["n_repeats"] == 2
    assert contrast["value"] == pytest.approx(.05)


def test_shared_runner_uses_observed_only_and_resumes(data, tmp_path, monkeypatch):
    import train

    class PolicyStub:
        def __init__(self, config, mode, seed):
            self.density_cache = object()
            assert config["algorithm"]["q_hidden"] == [100, 100]

        def fit(self, observed, density_cache=None):
            assert list(observed) == ["sample_id", "x1", "x2", "price", "inventory", "sales", "delta"]
            self.fits.append(observed.copy())
            return self

        def predict_price(self, x1, x2, inventory):
            return np.full(len(x1), .5)

        def predict_censored_demand(self, observed):
            return observed.loc[observed.delta == 0, "sales"].to_numpy()

        def predict_q(self, observed):
            return observed.price.to_numpy() * observed.sales.to_numpy()

    PolicyStub.fits = []
    monkeypatch.setattr(train, "PricingPipeline", PolicyStub)
    output = tmp_path / "results"
    train.train_shared(data.root, output, ["S16"], limit=1, resume=False)
    assert len(PolicyStub.fits) == 2
    pd.testing.assert_frame_equal(*PolicyStub.fits)
    train.train_shared(data.root, output, ["S16"], limit=1, resume=True)
    assert len(PolicyStub.fits) == 2
    rows = pd.read_csv(output / "metrics_per_run.csv")
    assert rows.dataset_sha256.unique().tolist() == [data.manifest_sha256]
    assert set(rows.method) == {"naive", "censoring_aware", "oracle"}

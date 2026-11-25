from pathlib import Path

import numpy as np
import pytest
from scipy.stats import poisson

from pricing_sim.data_store import freeze_data, load_config, load_run
from pricing_sim.learning import PricingPipeline
from pricing_sim.simulation import expected_profit, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "baseline.yaml")


def test_generator_is_seeded_and_uses_strict_censoring():
    observed, truth = generate_dataset(CONFIG, 22, 600)
    repeated, _ = generate_dataset(CONFIG, 22, 600)
    assert observed.equals(repeated)
    assert (truth.sales == np.minimum(truth.true_demand, truth.inventory)).all()
    assert (observed.delta == (truth.true_demand < truth.inventory).astype(int)).all()


def test_expected_profit_matches_poisson_sum():
    x1, x2, y, p = .2, .7, 4, .6
    rate = 5+x1*x1+x2*x2-5*p
    d = np.arange(1, 80)
    expected = np.sum(poisson.pmf(d-1, rate) *
                      (p*np.minimum(d, y)-.1*np.maximum(d-y, 0)))
    assert np.isclose(expected_profit(x1, x2, y, p, CONFIG), expected, atol=1e-10)


def test_frozen_data_cannot_be_overwritten(tmp_path):
    path = tmp_path / "v1"
    path.mkdir()
    with pytest.raises(FileExistsError):
        freeze_data(ROOT / "config" / "baseline.yaml", path)


def test_training_interface_rejects_truth():
    observed, test = load_run(ROOT / "data" / "v1", 0)
    assert len(observed) == len(test) == 2000
    assert "true_demand" not in observed.columns
    _, _, truth, _ = load_run(ROOT / "data" / "v1", 0, include_truth=True)
    with pytest.raises(ValueError):
        PricingPipeline(CONFIG, "naive").fit(truth)

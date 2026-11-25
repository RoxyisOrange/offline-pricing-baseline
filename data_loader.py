"""Public readers for the shared release and the separate E0 archive."""

from pathlib import Path

from pricing_sim.experiment_data import EXPERIMENTS, ExperimentData

DATA_ROOT = Path(__file__).resolve().parent / "data"


def open_dataset(data_dir=DATA_ROOT, *, scenario_id=None):
    """Open the verified release for repeated training, prediction and evaluation."""
    root = Path(data_dir)
    if scenario_id == "E0" and (root / "e0_frozen").is_dir():
        root = root / "e0_frozen"
    return ExperimentData(root)


def load_train(scenario_id, rep, *, data_dir=DATA_ROOT):
    """Return the seven observations accepted by the training pipeline."""
    return open_dataset(data_dir, scenario_id=scenario_id).training(scenario_id, rep)


def load_test(rep, *, scenario_id=None, data_dir=DATA_ROOT):
    """Read this round's shared test; pass scenario_id='E0' for the E0 archive."""
    return open_dataset(data_dir, scenario_id=scenario_id).test(rep)


def get_scenarios(experiment):
    """Return the scenario IDs in the experimental plan."""
    return ["E0"] if experiment == "E0" else list(EXPERIMENTS[experiment])

"""Freeze the E0 dataset or the shared E1–E4 experiment data."""
import argparse
from pathlib import Path
import subprocess

from pricing_sim.data_release import export_e0
from pricing_sim.data_store import freeze_data
from pricing_sim.experiment_data import EXPERIMENTS, N_ROUNDS, SCENARIOS, export_data, self_test

ROOT = Path(__file__).resolve().parent


def git_revision(ref: str) -> str | None:
    if not (ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", ref],
                                capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--experiment", choices=("all", *EXPERIMENTS),
                           help="all exports the 18 unique E1–E4 scenarios")
    selection.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS))
    selection.add_argument("--self-test", action="store_true", help="Check the shared generator")
    selection.add_argument("--export-e0", action="store_true", help="Copy the original E0 into a frozen CSV archive")
    parser.add_argument("--config", type=Path, help="E0 YAML configuration")
    parser.add_argument("--output", type=Path, help="New dataset directory")
    parser.add_argument("--rounds", type=int, choices=range(1, N_ROUNDS + 1),
                        metavar="1–100", help="Extension rounds to generate or check")
    args = parser.parse_args()
    if args.self_test:
        if args.config or args.output:
            parser.error("--self-test accepts only --rounds")
        self_test(args.rounds or N_ROUNDS)
    elif args.export_e0:
        if args.config or args.rounds is not None:
            parser.error("--export-e0 always copies the full archived E0")
        output = args.output or ROOT / "data" / "e0_frozen"
        manifest = export_e0(ROOT / "data" / "v1", output, source_revision=git_revision("baseline-v1^{commit}"))
        print(f"Archived {manifest['rounds']} original E0 pairs at {output}")
    elif args.experiment or args.scenarios:
        if args.output is None or args.config is not None:
            parser.error("Shared data needs --output and uses the fixed scenario settings")
        scenarios = args.scenarios or EXPERIMENTS.get(args.experiment, tuple(SCENARIOS))
        manifest = export_data(args.output, args.rounds or N_ROUNDS, scenarios, code_revision=git_revision("HEAD"))
        print(f"Frozen {len(manifest['scenarios']) * manifest['rounds']} training sets and "
              f"{manifest['rounds']} shared tests at {args.output}")
    else:
        if args.rounds is not None:
            parser.error("For E0, set the number of repeats in its --config file")
        output = args.output or ROOT / "data" / "v1"
        result = freeze_data(args.config or ROOT / "config" / "baseline.yaml", output)
        print(f"Frozen {result['n_repeats']} paired datasets at {output}")


if __name__ == "__main__":
    main()

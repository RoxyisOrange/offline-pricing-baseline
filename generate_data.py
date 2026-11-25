"""One-time generation of the frozen 100-run dataset."""
import argparse
from pathlib import Path
from pricing_sim.data_store import freeze_data

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "baseline.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "v1")
    args = parser.parse_args()
    result = freeze_data(args.config, args.output)
    print(f"Frozen {result['n_repeats']} paired datasets at {args.output}")


if __name__ == "__main__":
    main()

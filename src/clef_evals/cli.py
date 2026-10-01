"""CLI for clef-evals."""
import argparse
import json

from . import ClefJudge


def main():
    p = argparse.ArgumentParser(prog="clef-eval", description="Evaluate Clef as judge")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    print("See README for usage examples.")


if __name__ == "__main__":
    main()

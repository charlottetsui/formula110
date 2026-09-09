#!/usr/bin/env python3
"""Collect expert demonstrations, fit a clone, or evaluate autonomous driving."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.imitation import collect, evaluate, fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--seeds", type=int, nargs="+", required=True)
    collection.add_argument("--seconds", type=float, default=30.0)
    training = commands.add_parser("fit")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--validation-seeds", type=int, nargs="+", required=True)
    training.add_argument("--epochs", type=int, default=80)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--hidden-size", type=int, default=128)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[110, 42, 7, 2024, 8675309])
    evaluation.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.seeds, args.seconds)
    elif args.command == "fit":
        fit(args.dataset, args.output, args.validation_seeds, args.epochs, args.seed, args.hidden_size)
    else:
        evaluate(args.model, args.output, args.seeds, args.seconds)


if __name__ == "__main__":
    main()

"""CLI entrypoint for AFM04 prestage2."""

from __future__ import annotations

import argparse

from AFM04.prestage2.config import default_config
from AFM04.prestage2.runner import run_driver, run_prestage2_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--layer", type=str, default="layer_a")
    parser.add_argument("--assignment-path", type=str, default=None)
    parser.add_argument("--layer-a-result-path", type=str, default=None)
    args = parser.parse_args()

    cfg = default_config()
    if args.driver:
        result = run_driver(cfg)
    else:
        result = run_prestage2_shard(
            cfg,
            shard_index=args.shard_index,
            layer_name=args.layer,
            assignment_path=args.assignment_path,
            layer_a_result_path=args.layer_a_result_path,
        )
    print("prestage2 completed")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

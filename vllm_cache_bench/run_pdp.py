# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio

from run_nips import main


def parse_csv(value, cast):
    return [cast(item) for item in value.split(",") if item]


def build_pdp_config(args) -> str:
    config = {
        "pdp_initial_pd": args.initial_pd,
        "pdp_max_distance": args.max_distance,
        "pdp_recompute_interval": args.recompute_interval,
        "pdp_bucket_size": args.bucket_size,
        "pdp_eviction_distance": args.eviction_distance,
    }
    return ",".join(f"{key}={value}" for key, value in config.items())


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Run only the PDP prefix-cache eviction benchmark.")
    parser.add_argument("--datasets", default="sharegpt")
    parser.add_argument("--sizes", default="8000")
    parser.add_argument("--scales", default="1")
    parser.add_argument("--tag", default="pdp++")
    parser.add_argument("--initial-pd", type=int, default=32)
    parser.add_argument("--max-distance", type=int, default=256)
    parser.add_argument("--recompute-interval", type=int, default=512)
    parser.add_argument("--bucket-size", type=int, default=1)
    parser.add_argument("--eviction-distance", type=int, default=1)
    args = parser.parse_args()

    sizes = parse_csv(args.sizes, int)
    scales = parse_csv(args.scales, float)
    pdp_config = build_pdp_config(args)
    server_extra_args = f" --eviction_algorithm_config {pdp_config} "

    for dataset in parse_csv(args.datasets, str):
        asyncio.run(
            main(
                sizes,
                scales,
                "pdp",
                dataset,
                args.tag,
                client_algorithms=["pdp"],
                server_extra_args=server_extra_args,
            ))


if __name__ == "__main__":
    run()

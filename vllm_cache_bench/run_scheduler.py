import argparse
import asyncio
import re

from constants_nips import MODEL
from run_nips import main


def parse_csv(value, cast):
    return [cast(item) for item in value.split(",") if item]


def infer_model_size_b(model_name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    return float(match.group(1)) if match else 7.0


def build_scheduler_config(args) -> str:
    config = {
        "enable_scheduler": 1,
        "scheduler_warmup": args.warmup,
        "scheduler_min_events": args.min_events,
        "scheduler_small_threshold": args.small_threshold,
        "scheduler_large_threshold": args.large_threshold,
        "scheduler_observe_stride": args.observe_stride,
        "scheduler_shadow_policies": "|".join(args.shadow_policies),
        "model_size_b": infer_model_size_b(MODEL),
        "scheduler_initial_policy": args.initial_policy,
        "pdp_initial_pd": args.pdp_initial_pd,
        "pdp_max_distance": args.pdp_max_distance,
        "pdp_recompute_interval": args.pdp_recompute_interval,
        "pdp_bucket_size": args.pdp_bucket_size,
        "pdp_eviction_distance": args.pdp_eviction_distance,
    }
    return ",".join(f"{key}={value}" for key, value in config.items())


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Run only the dynamic eviction-policy scheduler benchmark.")
    parser.add_argument("--datasets", default="sharegpt,lmsys,chatbot")
    parser.add_argument("--sizes", default="8000")
    parser.add_argument("--scales", default="1")
    parser.add_argument("--tag", default="scheduler++")
    parser.add_argument("--warmup", type=float, default=100)
    parser.add_argument("--min-events", type=int, default=0)
    parser.add_argument("--observe-stride", type=int, default=4)
    parser.add_argument("--small-threshold", type=float, default=0.10)
    parser.add_argument("--large-threshold", type=float, default=0.05)
    parser.add_argument("--initial-policy",
                        choices=("ml", "lru", "rrip", "fifo", "pdp"),
                        default="ml")
    parser.add_argument(
        "--shadow-policies",
        default="lru,rrip,fifo,pdp",
        help=("Comma-separated non-ML policies to shadow during warmup. "
              "Use lru,rrip,fifo for the original scheduler or include pdp "
              "for PDP-aware runs."))
    parser.add_argument("--pdp-initial-pd", type=int, default=32)
    parser.add_argument("--pdp-max-distance", type=int, default=256)
    parser.add_argument("--pdp-recompute-interval", type=int, default=512)
    parser.add_argument("--pdp-bucket-size", type=int, default=1)
    parser.add_argument("--pdp-eviction-distance", type=int, default=1)
    args = parser.parse_args()
    args.shadow_policies = parse_csv(args.shadow_policies, str)

    sizes = parse_csv(args.sizes, int)
    scales = parse_csv(args.scales, float)
    scheduler_config = build_scheduler_config(args)
    server_extra_args = (
        f" --eviction_algorithm_config {scheduler_config} ")

    for dataset in parse_csv(args.datasets, str):
        asyncio.run(
            main(
                sizes,
                scales,
                "ml",
                dataset,
                args.tag,
                client_algorithms=["scheduler"],
                server_extra_args=server_extra_args,
                post_warmup_seconds=args.warmup,
            ))


if __name__ == "__main__":
    run()

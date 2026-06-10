import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_LOG_GLOB = "*/**/client_logs/*.json"
DEFAULT_POLICIES = ("lru", "scheduler", "ml", "rrip", "fifo", "pdp")


def percentile(values: list[int], percentile_value: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile_value)
    return sorted_values[index]


def infer_policy(path: Path, record: dict) -> Optional[str]:
    algorithm = record.get("algorithm")
    if isinstance(algorithm, str):
        return algorithm
    stem = path.stem
    for policy in DEFAULT_POLICIES:
        if stem.endswith(f"_{policy}") or stem == policy:
            return policy
    return None


def infer_workload(path: Path, record: dict) -> str:
    dataset_name = record.get("dataset_name")
    if isinstance(dataset_name, str) and dataset_name:
        return dataset_name
    for part in path.parts:
        if "-" in part:
            workload = part.split("-", 1)[0]
            if workload in ("sharegpt", "lmsys", "chatbot", "tay"):
                return workload
    return "unknown"


def infer_model(path: Path, record: dict) -> str:
    model_id = record.get("model_id")
    if isinstance(model_id, str) and model_id:
        return model_id.rsplit("/", 1)[-1]
    for part in path.parts:
        if "Instruct" in part or part.startswith("Qwen"):
            return part
    return "unknown"


def iter_log_records(results_dir: Path) -> Iterable[tuple[Path, dict]]:
    for path in sorted(results_dir.glob(DEFAULT_LOG_GLOB)):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("input_lens") and record.get("output_lens"):
            yield path, record


def summarize_record(path: Path, record: dict) -> dict:
    input_lens = [int(value) for value in record["input_lens"]]
    output_lens = [int(value) for value in record["output_lens"]]
    return {
        "model": infer_model(path, record),
        "workload": infer_workload(path, record),
        "policy": infer_policy(path, record) or "unknown",
        "completed": int(record.get("completed", len(input_lens))),
        "avg_prompt_tokens": statistics.mean(input_lens),
        "p50_prompt": percentile(input_lens, 0.50),
        "p95_prompt": percentile(input_lens, 0.95),
        "avg_response_tokens": statistics.mean(output_lens),
        "p50_response": percentile(output_lens, 0.50),
        "p95_response": percentile(output_lens, 0.95),
        "source": str(path),
    }


def select_rows(rows: list[dict], preferred_policies: list[str]) -> list[dict]:
    by_workload = {}
    policy_rank = {policy: rank for rank, policy in enumerate(preferred_policies)}
    for row in rows:
        key = (row["model"], row["workload"])
        current = by_workload.get(key)
        if current is None:
            by_workload[key] = row
            continue
        current_rank = policy_rank.get(current["policy"], len(policy_rank))
        row_rank = policy_rank.get(row["policy"], len(policy_rank))
        if (row_rank, -row["completed"], row["source"]) < (
                current_rank, -current["completed"], current["source"]):
            by_workload[key] = row
    return [by_workload[key] for key in sorted(by_workload)]


def print_markdown(rows: list[dict]) -> None:
    print("| model | workload | policy | completed requests | avg prompt tokens | "
          "p50 prompt | p95 prompt | avg response tokens | p50 response | "
          "p95 response |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['model']} | {row['workload']} | {row['policy']} | "
            f"{row['completed']} | {row['avg_prompt_tokens']:.1f} | "
            f"{row['p50_prompt']} | {row['p95_prompt']} | "
            f"{row['avg_response_tokens']:.1f} | {row['p50_response']} | "
            f"{row['p95_response']} |")


def print_csv(rows: list[dict]) -> None:
    headers = (
        "model",
        "workload",
        "policy",
        "completed",
        "avg_prompt_tokens",
        "p50_prompt",
        "p95_prompt",
        "avg_response_tokens",
        "p50_response",
        "p95_response",
        "source",
    )
    print(",".join(headers))
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.1f}"
            values.append(str(value))
        print(",".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize prompt and response token lengths from benchmark "
        "client logs.")
    parser.add_argument("--results-dir",
                        type=Path,
                        default=Path("results"),
                        help="Directory containing benchmark result folders.")
    parser.add_argument("--policies",
                        default="lru,scheduler,ml,rrip,fifo,pdp",
                        help="Preferred policies when multiple logs exist for "
                        "the same model/workload.")
    parser.add_argument("--format",
                        choices=("markdown", "csv"),
                        default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preferred_policies = [
        policy for policy in args.policies.split(",") if policy
    ]
    rows = [
        summarize_record(path, record)
        for path, record in iter_log_records(args.results_dir)
    ]
    rows = select_rows(rows, preferred_policies)
    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()

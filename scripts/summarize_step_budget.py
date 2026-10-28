"""Summarize planned step-budget runs from local ledgers, without API calls or simulator access."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(directory):
    experiment = json.loads((directory / "experiment.json").read_text())
    records = []
    tasks = experiment.get("tasks", [experiment.get("task", "ButtonUnmask")])
    planned = [(task, episode) for task in tasks for episode in experiment["episodes"]]
    for task, episode in planned:
        artifacts = directory / f"{task.lower()}-ep{episode}" / "artifacts"
        path = artifacts / "agent/report.json"
        if not path.exists():
            raise ValueError(f"run {episode} has no final report; do not label an unfinished run as completed")
        report = json.loads(path.read_text())
        metrics = report["metrics"]
        schedule = metrics["scheduling_accounting"]
        tokens = metrics["token_accounting"]
        result = report["result"]
        task_commit = artifacts / "experience-task-commit.json"
        global_commit = artifacts / "experience-commit.json"
        records.append(
            {
                "task": task,
                "episode": episode,
                "outcome": result["outcome"],
                "success": result.get("feedback", {}).get("success"),
                "control_steps": result["control_steps"],
                "api_calls": report["api_calls"],
                "execution_api_calls": metrics["api_execution_calls"],
                "review_api_calls": metrics["api_review_calls"],
                "api_failures": metrics["api_failures"],
                "api_retries": metrics["api_retries"],
                "input_tokens": report["input_tokens"],
                "cached_input_tokens": metrics["cached_input_tokens"],
                "uncached_input_tokens": tokens["uncached_input_tokens"],
                "output_tokens": report["output_tokens"],
                "total_tokens": tokens["total_tokens"],
                "usage_missing_responses": tokens["usage_missing_response_count"],
                "episode_wall_seconds": metrics["episode_wall_seconds"],
                "run_wall_seconds": metrics["run_wall_seconds"],
                "api_seconds": metrics["api_seconds"],
                "vla_inference_seconds": metrics["vla_inference_seconds"],
                "intervention_steps": metrics["execution_accounting"]["gpt_control_steps"],
                "delegated_steps": schedule["executed_steps"],
                "contracts": schedule["contract_count"],
                "zero_step_reports": schedule["zero_step_reports"],
                "avoided_chunk_review_equivalents": schedule["avoided_chunk_review_equivalents"],
                "review_equivalent_reduction_fraction": schedule["review_equivalent_reduction_fraction"],
                "guard_triggers": schedule["guard_trigger_counts"],
                "response_categories": schedule["response_categories"],
                "tokens_by_phase": schedule["tokens_by_phase"],
                "astra_delivered_frame_count": schedule["astra_delivered_frame_count"],
                "verified": report["verified"],
                "error_type": report["error_type"],
                "stop_reason": report["stop_reason"],
                "review_written": report["review_written_by_model"],
                "task_experience_commit": json.loads(task_commit.read_text()) if task_commit.exists() else None,
                "global_experience_commit": json.loads(global_commit.read_text()) if global_commit.exists() else None,
                "input_experience": metrics["experience"],
                "source_hashes": json.loads((artifacts / "agent/source-hashes.json").read_text()),
                "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "report_path": str(path),
            }
        )
    summary = {
        "experiment": experiment,
        "runs": records,
        "limitations": "New-mode episodes with experience updates (sequential per task), no matched old API baseline. "
        "Review equivalents fix the actual trajectory and do not estimate saved tokens or establish causal speedup. "
        "Sparse-history delivery and long scheduling both affect cost. Unlabelled visual misses are unknown, not zero.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    flat_keys = [
        key
        for key, value in records[0].items()
        if not isinstance(value, dict | list) and key not in ("task_experience_commit", "global_experience_commit")
    ]
    with (directory / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.experiment_dir)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "task",
                        "episode",
                        "outcome",
                        "api_calls",
                        "total_tokens",
                        "control_steps",
                        "avoided_chunk_review_equivalents",
                        "task_experience_commit",
                    )
                }
                for r in summary["runs"]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

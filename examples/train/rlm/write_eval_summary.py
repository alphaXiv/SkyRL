"""Write EVAL.md after run_rlm_eval.sh completes.

Reads aggregated_results.jsonl for eval metrics and fetches GPU/memory stats
from the most recent matching wandb run.
"""

import argparse
import json
import sys
from pathlib import Path


def _format_duration(total_seconds: float) -> str:
    h, rem = divmod(int(total_seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _fetch_wandb_system_metrics(project: str, run_name: str):
    """Return (avg_gpu_util_pct, avg_gpu_mem_pct) from the most recent matching run."""
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(project, filters={"display_name": run_name}, order="-created_at")
        run = next(iter(runs), None)
        if run is None:
            print(f"Warning: no wandb run named '{run_name}' found in project '{project}'", file=sys.stderr)
            return None, None

        system_df = run.history(stream="system")
        if system_df.empty:
            print("Warning: wandb system metrics are empty", file=sys.stderr)
            return None, None

        # GPU utilization: columns like "system.gpu.0.gpu", "system.gpu.1.gpu", ...
        util_cols = [c for c in system_df.columns if c.startswith("system.gpu.") and c.endswith(".gpu")]
        # GPU memory: memoryUsed / memoryTotal * 100 per device, then average
        mem_used_cols = [c for c in system_df.columns if c.startswith("system.gpu.") and c.endswith(".memoryUsed")]
        mem_total_cols = [c for c in system_df.columns if c.startswith("system.gpu.") and c.endswith(".memoryTotal")]

        avg_gpu_util = None
        if util_cols:
            avg_gpu_util = system_df[util_cols].mean().mean()

        avg_gpu_mem_pct = None
        if mem_used_cols and mem_total_cols:
            used = system_df[mem_used_cols].mean().mean()
            total = system_df[mem_total_cols].mean().mean()
            if total > 0:
                avg_gpu_mem_pct = used / total * 100

        return avg_gpu_util, avg_gpu_mem_pct
    except Exception as e:
        print(f"Warning: could not fetch wandb system metrics: {e}", file=sys.stderr)
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-path", required=True, help="trainer.export_path from the eval run")
    parser.add_argument("--wandb-project", default="rlm")
    parser.add_argument("--wandb-run-name", default="rlm_eval")
    parser.add_argument("--output", default="EVAL.md", help="Path to write the summary markdown")
    args = parser.parse_args()

    results_path = Path(args.export_path) / "dumped_evals" / "eval_only" / "aggregated_results.jsonl"
    if not results_path.exists():
        print(f"Error: results file not found at {results_path}", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        metrics = json.loads(f.readline())

    # Wall-clock time from first generate() call to last — excludes startup, data loading, etc.
    total_generation_time_s = metrics.get("eval/all/total_generation_time_s")

    # per_trajectory_latency_s = total_generate_time / #completed_trajectories (step-wise runs).
    # Falls back to rollout_latency_s (= time / total steps) for non-step-wise runs.
    per_request_latency_s = metrics.get("eval/all/per_trajectory_latency_s") or metrics.get("eval/all/rollout_latency_s")

    avg_gpu_util, avg_gpu_mem_pct = _fetch_wandb_system_metrics(args.wandb_project, args.wandb_run_name)

    lines = [
        "# RLM Eval Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if total_generation_time_s is not None:
        lines.append(f"| Total generation latency | {_format_duration(total_generation_time_s)} ({total_generation_time_s:.0f}s) |")
    else:
        lines.append("| Total generation latency | N/A |")

    if per_request_latency_s is not None:
        lines.append(f"| Avg per-request completion latency | {per_request_latency_s:.3f}s |")
    else:
        lines.append("| Avg per-request completion latency | N/A |")

    if avg_gpu_util is not None:
        lines.append(f"| Avg GPU utilization | {avg_gpu_util:.1f}% |")
    else:
        lines.append("| Avg GPU utilization | N/A (wandb unavailable) |")

    if avg_gpu_mem_pct is not None:
        lines.append(f"| Avg GPU memory utilization | {avg_gpu_mem_pct:.1f}% |")
    else:
        lines.append("| Avg GPU memory utilization | N/A (wandb unavailable) |")

    lines += [
        "",
        "## All Eval Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in sorted(metrics.items()):
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| `{k}` | {val_str} |")

    output = Path(args.output)
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

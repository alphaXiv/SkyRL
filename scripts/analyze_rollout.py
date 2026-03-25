#!/usr/bin/env python3
"""Pretty-print rollout data from a dumped eval JSONL file."""

import argparse
import json
import textwrap
from pathlib import Path

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
}


def c(text: str, *styles: str) -> str:
    prefix = "".join(COLORS.get(s, "") for s in styles)
    return f"{prefix}{text}{COLORS['reset']}"


def hr(char="─", width=88) -> str:
    return c(char * width, "dim")


def section_header(title: str) -> str:
    pad = 2
    inner = f" {title} "
    side = (88 - len(inner) - 2 * pad) // 2
    line = "─" * side
    return c(f"{'─' * pad}{line}┤ ", "dim") + c(title, "bold", "cyan") + c(f" ├{line}{'─' * pad}", "dim")


def parse_prompt_turns(prompt: str) -> list[dict]:
    """Split a ChatML-formatted prompt into turns."""
    turns = []
    parts = prompt.split("<|im_start|>")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        role_end = part.find("\n")
        if role_end == -1:
            role = part.rstrip("<|im_end|>").strip()
            content = ""
        else:
            role = part[:role_end].strip()
            content = part[role_end + 1:]
        content = content.replace("<|im_end|>", "").strip()
        turns.append({"role": role, "content": content})
    return turns


def format_role(role: str) -> str:
    role_styles = {
        "system": ("magenta", "bold"),
        "user": ("green", "bold"),
        "assistant": ("blue", "bold"),
    }
    styles = role_styles.get(role, ("white", "bold"))
    return c(f"  [{role.upper()}]", *styles)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = (max_chars - 40) // 2
    return text[:half] + c(f"\n  ... ({len(text) - max_chars:,} chars truncated) ...\n", "dim", "yellow") + text[-half:]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~3.5 chars per token for English + markup."""
    return max(1, int(len(text) / 3.5))


def _fmt_scores(scores: list) -> str:
    """Format a score list as a colored string, showing the actual value."""
    if not scores:
        return c("no scores", "dim")
    avg = sum(scores) / len(scores)
    val_str = f"{avg:.3f}" if len(scores) == 1 else f"avg {avg:.3f}"
    if avg >= 0.99:
        return c(f"{val_str}  (exact match)", "bold", "green")
    elif avg > 0.0:
        return c(f"{val_str}  (partial)", "yellow", "bold")
    else:
        return c(f"{val_str}  (no match)", "bold", "red")


def print_step_wise_trajectory(traj_idx: int, steps: list[dict], max_response_chars: int, max_context_chars: int, show_context: bool):
    """Print a grouped step-wise trajectory (multiple records = one trajectory)."""
    first = steps[0]
    last = steps[-1]
    extras = first["env_extras"]
    reward_spec = extras.get("reward_spec", {})
    if isinstance(reward_spec, str):
        reward_spec = eval(reward_spec)
    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = eval(extra_info)

    ground_truth = reward_spec.get("ground_truth", "N/A")
    max_turns = extras.get("max_turns", "N/A")
    context_text = extra_info.get("context_text", "")
    env_class = first["env_class"]
    data_source = first["data_source"]
    scores = last["score"]
    stop_reason = last["stop_reason"]

    # Combine all steps into the full conversation
    full_transcript = first["input_prompt"]
    for step in steps:
        full_transcript += step["output_response"]
    turns = parse_prompt_turns(full_transcript)
    assistant_turns = [t for t in turns if t["role"] == "assistant" and len(t["content"].strip()) > 10]
    repl_turns = [t for t in turns if t["role"] == "user" and t["content"].startswith("Code executed:")]

    total_response_chars = sum(len(s["output_response"]) for s in steps)
    total_response_tokens = sum(estimate_tokens(s["output_response"]) for s in steps)

    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ║", "bold", "cyan") + c(f"{'TRAJECTORY #' + str(traj_idx):^86}", "bold", "white") + c("║", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ╚{'═' * 86}╝", "bold", "cyan"))
    print()

    print(section_header("Metadata"))
    print()

    meta_rows = [
        ("Env Class", c(env_class, "bold")),
        ("Data Source", data_source),
        ("Stop Reason", c(stop_reason, "green" if stop_reason == "stop" else "yellow", "bold")),
        ("Ground Truth", c(str(ground_truth), "bold", "green")),
        ("Max Turns", str(max_turns)),
        ("Actual Steps", f"{len(steps)} steps, {len(assistant_turns)} assistant, {len(repl_turns)} REPL"),
        ("Total Generation", f"~{total_response_tokens:,} tokens ({total_response_chars:,} chars)"),
        ("Score Samples", f"{len(scores):,}"),
        ("Score (F1)", _fmt_scores(scores)),
    ]
    for label, value in meta_rows:
        print(f"  {c(label + ':', 'bold'):>38s}  {value}")

    print()
    print(section_header("Per-Step Lengths"))
    print()
    print(f"  {'Step':>6}  {'Prompt':>14}  {'Response':>14}  {'Cumulative':>14}  {'Stop':>8}")
    print(f"  {'─' * 6}  {'─' * 14}  {'─' * 14}  {'─' * 14}  {'─' * 8}")

    cumulative_tokens = 0
    for si, step in enumerate(steps):
        prompt_tok = estimate_tokens(step["input_prompt"])
        resp_tok = estimate_tokens(step["output_response"])
        cumulative_tokens += resp_tok
        sr = step["stop_reason"]
        sr_colored = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
        print(f"  {si + 1:>6}  {prompt_tok:>8} tok  {resp_tok:>8} tok  {cumulative_tokens:>8} tok  {sr_colored:>19}")

    print()
    init_prompt_tok = estimate_tokens(first["input_prompt"])
    final_prompt_tok = estimate_tokens(last["input_prompt"])
    print(f"  {c('Initial prompt:', 'dim')}  ~{init_prompt_tok:,} tokens ({len(first['input_prompt']):,} chars)")
    print(f"  {c('Final prompt:', 'dim')}   ~{final_prompt_tok:,} tokens ({len(last['input_prompt']):,} chars)")
    print(f"  {c('Context size:', 'dim')}   {len(context_text):,} chars")
    print()

    print(section_header("Conversation"))

    user_query = None
    for turn in turns:
        if turn["role"] == "user" and not turn["content"].startswith("[Execution Result]"):
            user_query = turn["content"]

    if user_query:
        print()
        print(f"  {c('Query:', 'bold', 'yellow')}  {user_query}")

    print()
    for turn in turns:
        print(format_role(turn["role"]))
        for line in turn["content"].split("\n"):
            print(f"    {line}")
        print()

    if show_context and context_text:
        print(section_header("Context (from extra_info)"))
        print()
        ctx = truncate(context_text, max_context_chars)
        for line in ctx.split("\n"):
            print(f"    {line}")
        print()

    print(hr())
    print()


def print_rollout(idx: int, record: dict, max_response_chars: int, max_context_chars: int, show_context: bool):
    extras = record["env_extras"]
    reward_spec = extras.get("reward_spec", {})
    if isinstance(reward_spec, str):
        reward_spec = eval(reward_spec)
    extra_info = extras.get("extra_info", {})
    if isinstance(extra_info, str):
        extra_info = eval(extra_info)

    ground_truth = reward_spec.get("ground_truth", "N/A")
    max_turns = extras.get("max_turns", "N/A")
    context_text = extra_info.get("context_text", "")

    prompt = record["input_prompt"]
    response = record["output_response"]
    scores = record["score"]
    stop_reason = record["stop_reason"]
    env_class = record["env_class"]
    data_source = record["data_source"]

    full_transcript = prompt + response
    turns = parse_prompt_turns(full_transcript)

    assistant_turns = [t for t in turns if t["role"] == "assistant" and len(t["content"].strip()) > 10]
    repl_turns = [t for t in turns if t["role"] == "user" and t["content"].startswith("Code executed:")]

    prompt_tok = estimate_tokens(prompt)
    resp_tok = estimate_tokens(response)

    print()
    print(c(f"  ╔{'═' * 86}╗", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ║", "bold", "cyan") + c(f"{'ROLLOUT #' + str(idx):^86}", "bold", "white") + c("║", "bold", "cyan"))
    print(c(f"  ║{'':^86}║", "bold", "cyan"))
    print(c(f"  ╚{'═' * 86}╝", "bold", "cyan"))
    print()

    print(section_header("Metadata"))
    print()

    meta_rows = [
        ("Env Class", c(env_class, "bold")),
        ("Data Source", data_source),
        ("Stop Reason", c(stop_reason, "green" if stop_reason == "stop" else "yellow")),
        ("Ground Truth", c(str(ground_truth), "bold", "green")),
        ("Max Turns", str(max_turns)),
        ("Agent Turns", f"{len(assistant_turns)} assistant, {len(repl_turns)} REPL"),
        ("Prompt Length", f"~{prompt_tok:,} tokens ({len(prompt):,} chars)"),
        ("Response Length", f"~{resp_tok:,} tokens ({len(response):,} chars)"),
        ("Score (F1)", _fmt_scores(scores)),
    ]
    for label, value in meta_rows:
        print(f"  {c(label + ':', 'bold'):>38s}  {value}")

    print()
    print(section_header("Conversation"))

    user_query = None
    for turn in turns:
        if turn["role"] == "user" and not turn["content"].startswith("[Execution Result]"):
            user_query = turn["content"]

    if user_query:
        print()
        print(f"  {c('Query:', 'bold', 'yellow')}  {user_query}")

    print()
    for turn_num, turn in enumerate(turns):
        print(format_role(turn["role"]))
        content = turn["content"]
        if turn["role"] == "system":
            pass
        elif turn["role"] == "user" and turn["content"].startswith("[Execution Result]"):
            pass

        for line in content.split("\n"):
            print(f"    {line}")
        print()

    if show_context and context_text:
        print(section_header("Context (from extra_info)"))
        print()
        ctx = truncate(context_text, max_context_chars)
        for line in ctx.split("\n"):
            print(f"    {line}")
        print()

    print(hr())
    print()


def group_step_wise_trajectories(records: list[dict]) -> list[list[dict]]:
    """Group step-wise records into trajectories.

    Step-wise records for the same trajectory have growing input_prompt lengths
    and share the same initial prompt. A new trajectory starts when the prompt
    length resets (shrinks or equals the first record's prompt length).
    """
    if not records:
        return []

    trajectories = []
    current: list[dict] = [records[0]]

    for r in records[1:]:
        prev_len = len(current[-1]["input_prompt"])
        cur_len = len(r["input_prompt"])
        if cur_len <= len(current[0]["input_prompt"]):
            trajectories.append(current)
            current = [r]
        else:
            current.append(r)
    if current:
        trajectories.append(current)
    return trajectories


def print_trajectory_summary(trajectories: list[list[dict]]):
    """Print a compact summary table of all trajectories."""
    print(f"  {'#':>4}  {'Steps':>5}  {'Stop':>8}  {'Init Prompt':>13}  {'Final Prompt':>13}  {'Tot Gen':>13}  {'Score':>8}  {'GT':>6}")
    print(f"  {'─' * 4}  {'─' * 5}  {'─' * 8}  {'─' * 13}  {'─' * 13}  {'─' * 13}  {'─' * 8}  {'─' * 6}")
    for ti, steps in enumerate(trajectories):
        n_steps = len(steps)
        sr = steps[-1]["stop_reason"]
        init_tok = estimate_tokens(steps[0]["input_prompt"])
        final_tok = estimate_tokens(steps[-1]["input_prompt"])
        total_gen_tok = sum(estimate_tokens(s["output_response"]) for s in steps)
        scores = steps[-1]["score"]
        score_str = _fmt_scores(scores)
        extras = steps[0]["env_extras"]
        rs = extras.get("reward_spec", {})
        if isinstance(rs, str):
            rs = eval(rs)
        gt = rs.get("ground_truth", "?")
        sr_colored = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
        print(f"  {ti:>4}  {n_steps:>5}  {sr_colored:>19}  {init_tok:>8} tok  {final_tok:>8} tok  {total_gen_tok:>8} tok  {score_str:>19}  {str(gt):>6}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and pretty-print rollouts from a dumped eval JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/analyze_rollout.py eval.jsonl
              python scripts/analyze_rollout.py eval.jsonl --max-rollouts 5
              python scripts/analyze_rollout.py eval.jsonl --show-context --max-context-chars 2000
              python scripts/analyze_rollout.py eval.jsonl --offset 3 --max-rollouts 2
              python scripts/analyze_rollout.py --summary          # compact table of all trajectories
        """),
    )
    parser.add_argument("file", type=Path, nargs="?", default=Path("/root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl"), help="Path to the JSONL file (default: /root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl)")
    parser.add_argument("--max-rollouts", type=int, default=1, help="Number of rollouts/trajectories to display (default: 1)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rollouts/trajectories before printing (default: 0)")
    parser.add_argument("--max-response-chars", type=int, default=3000, help="Max chars to show per response (default: 3000)")
    parser.add_argument("--show-context", action="store_true", help="Also print the context_text from extra_info")
    parser.add_argument("--max-context-chars", type=int, default=2000, help="Max chars of context to show (default: 2000)")
    parser.add_argument("--scores-only", action="store_true", help="Only print a compact score summary for each raw record")
    parser.add_argument("--summary", action="store_true", help="Print a compact trajectory summary table with per-step lengths")
    parser.add_argument("--raw", action="store_true", help="Show raw per-record view instead of grouped trajectory view")
    args = parser.parse_args()

    if not args.file.exists():
        print(c(f"Error: file not found: {args.file}", "red", "bold"))
        return

    records = []
    with open(args.file) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    trajectories = group_step_wise_trajectories(records)
    is_step_wise = len(trajectories) < total

    print()
    print(c(f"  Loaded {total} records from ", "dim") + c(str(args.file), "bold", "underline"))
    if is_step_wise:
        print(c(f"  Grouped into {len(trajectories)} trajectories (step-wise, {total / len(trajectories):.0f} steps avg)", "dim"))
    print()

    if args.scores_only:
        print(f"  {'#':>4}  {'Stop':>8}  {'GT':>6}  {'Env':>6}  {'Score':>12}  {'Prompt':>10}  {'Resp':>10}")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 12}  {'─' * 10}  {'─' * 10}")
        for i in range(total):
            r = records[i]
            extras = r["env_extras"]
            rs = extras.get("reward_spec", {})
            if isinstance(rs, str):
                rs = eval(rs)
            gt = rs.get("ground_truth", "?")
            score_str = _fmt_scores(r["score"])
            p_tok = estimate_tokens(r["input_prompt"])
            r_tok = estimate_tokens(r["output_response"])
            print(f"  {i:>4}  {r['stop_reason']:>8}  {str(gt):>6}  {r['env_class']:>6}  {score_str:>23}  {p_tok:>6} tok  {r_tok:>6} tok")
        print()
        return

    if args.summary:
        if is_step_wise:
            print_trajectory_summary(trajectories)
            for ti, steps in enumerate(trajectories):
                print(c(f"  Trajectory {ti}:", "bold"))
                print(f"    {'Step':>4}  {'Prompt':>10}  {'Response':>10}  {'Stop':>8}")
                print(f"    {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
                for si, s in enumerate(steps):
                    p = estimate_tokens(s["input_prompt"])
                    r = estimate_tokens(s["output_response"])
                    sr = s["stop_reason"]
                    sr_c = c(sr, "green") if sr == "stop" else c(sr, "yellow", "bold")
                    print(f"    {si + 1:>4}  {p:>6} tok  {r:>6} tok  {sr_c:>19}")
                print()
        else:
            for i in range(total):
                r = records[i]
                p = estimate_tokens(r["input_prompt"])
                resp = estimate_tokens(r["output_response"])
                print(f"  #{i}: prompt={p} tok  resp={resp} tok  stop={r['stop_reason']}")
            print()
        return

    if is_step_wise and not args.raw:
        start = args.offset
        end = min(start + args.max_rollouts, len(trajectories))
        if start >= len(trajectories):
            print(c(f"Offset {start} is beyond the {len(trajectories)} trajectories.", "red", "bold"))
            return
        print(c(f"  Showing trajectories {start}..{end - 1}  (of {len(trajectories)} total)", "dim"))
        print()
        for ti in range(start, end):
            print_step_wise_trajectory(ti, trajectories[ti], args.max_response_chars, args.max_context_chars, args.show_context)
    else:
        start = args.offset
        end = min(start + args.max_rollouts, total)
        if start >= total:
            print(c(f"Offset {start} is beyond the {total} records.", "red", "bold"))
            return
        print(c(f"  Showing rollouts {start}..{end - 1}  (of {total} total)", "dim"))
        print()
        for i in range(start, end):
            print_rollout(i, records[i], args.max_response_chars, args.max_context_chars, args.show_context)


if __name__ == "__main__":
    main()

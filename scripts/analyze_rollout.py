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

    unique_scores = set(scores)
    any_nonzero = any(s != 0.0 for s in scores)

    # The input_prompt has the initial turns (system, user, maybe assistant start).
    # The output_response has the model's reply which may contain further
    # <|im_start|>user (REPL feedback) and <|im_start|>assistant turns.
    # Combine them and parse once to get the full conversation.
    full_transcript = prompt + response
    turns = parse_prompt_turns(full_transcript)

    # Count assistant turns (excluding empty ones like a bare <think> tag)
    assistant_turns = [t for t in turns if t["role"] == "assistant" and len(t["content"].strip()) > 10]
    repl_turns = [t for t in turns if t["role"] == "user" and t["content"].startswith("[Execution Result]")]

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
        ("Response Length", f"~{len(response.split()):,} words ({len(response):,} chars)"),
        ("Score Samples", f"{len(scores):,}"),
        ("Score Values", c("PASS", "bold", "green") if any_nonzero else c(f"ALL ZERO  (unique: {unique_scores})", "bold", "red")),
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
        """),
    )
    parser.add_argument("file", type=Path, nargs="?", default=Path("/root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl"), help="Path to the JSONL file (default: /root/SkyRL/tmp/rlm-eval/dumped_evals/eval_only/unknown.jsonl)")
    parser.add_argument("--max-rollouts", type=int, default=1, help="Number of rollouts to display (default: 1)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rollouts before printing (default: 0)")
    parser.add_argument("--max-response-chars", type=int, default=3000, help="Max chars to show per response (default: 3000)")
    parser.add_argument("--show-context", action="store_true", help="Also print the context_text from extra_info")
    parser.add_argument("--max-context-chars", type=int, default=2000, help="Max chars of context to show (default: 2000)")
    parser.add_argument("--scores-only", action="store_true", help="Only print a compact score summary for each rollout")
    args = parser.parse_args()

    if not args.file.exists():
        print(c(f"Error: file not found: {args.file}", "red", "bold"))
        return

    records = []
    with open(args.file) as f:
        for line in f:
            records.append(json.loads(line))

    total = len(records)
    start = args.offset
    end = min(start + args.max_rollouts, total)

    if start >= total:
        print(c(f"Offset {start} is beyond the {total} records in the file.", "red", "bold"))
        return

    print()
    print(c(f"  Loaded {total} rollouts from ", "dim") + c(str(args.file), "bold", "underline"))
    print(c(f"  Showing rollouts {start}..{end - 1}  (of {total} total)", "dim"))
    print()

    if args.scores_only:
        print(f"  {'#':>4}  {'Stop':>8}  {'GT':>6}  {'Env':>6}  {'Score':>12}  {'Resp Len':>10}")
        print(f"  {'─' * 4}  {'─' * 8}  {'─' * 6}  {'─' * 6}  {'─' * 12}  {'─' * 10}")
        for i in range(start, end):
            r = records[i]
            extras = r["env_extras"]
            rs = extras.get("reward_spec", {})
            if isinstance(rs, str):
                rs = eval(rs)
            gt = rs.get("ground_truth", "?")
            unique_s = set(r["score"])
            nonzero = any(s != 0.0 for s in r["score"])
            score_str = c("PASS", "green") if nonzero else c("FAIL", "red")
            print(f"  {i:>4}  {r['stop_reason']:>8}  {str(gt):>6}  {r['env_class']:>6}  {score_str:>23}  {len(r['output_response']):>10,}")
        print()
        return

    for i in range(start, end):
        print_rollout(i, records[i], args.max_response_chars, args.max_context_chars, args.show_context)


if __name__ == "__main__":
    main()

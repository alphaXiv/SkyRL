"""
Create RLM dataset (parquet) from oolongbench/oolong-real on HuggingFace.

Each example has two text inputs:
- **prompt**: the question the model must answer (from the ``question`` column).
- **context_text** (in extra_info): large external context stored in the REPL
  as ``context`` (from the ``context_window_text`` column).

The model must set `Final` to match ground_truth (the ``answer`` column) to get
reward.
"""

import argparse
import os

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/rlm")
    parser.add_argument("--n_train", type=int, default=1, help="Cap training examples")
    parser.add_argument("--n_val", type=int, default=1, help="Cap validation examples")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--subset", default="dnd", help="Dataset subset (e.g. dnd, toy_dnd)")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    hf_ds = datasets.load_dataset("oolongbench/oolong-real", args.subset)

    def convert(row: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": row["question"]}],
            "env_class": "rlm",
            "reward_spec": {"ground_truth": row["answer"]},
            "max_turns": args.max_turns,
            "extra_info": {"context_text": row["context_window_text"]},
        }

    def convert_split(raw: datasets.Dataset, cap: int | None) -> datasets.Dataset:
        n = min(cap, len(raw)) if cap is not None else len(raw)
        rows = [convert(raw[i]) for i in range(n)]
        return datasets.Dataset.from_list(rows)

    splits: dict[str, datasets.Dataset] = {}
    if "validation" in hf_ds:
        splits["validation"] = convert_split(hf_ds["validation"], args.n_val)

    if "test" in hf_ds:
        splits["train"] = convert_split(hf_ds["test"], args.n_train)
    elif "train" in hf_ds:
        splits["train"] = convert_split(hf_ds["train"], args.n_train)

    n_show = 3
    for split_name, ds in splits.items():
        print(f"\nFirst {n_show} {split_name} examples:")
        for i in range(min(n_show, len(ds))):
            ex = ds[i]
            print(f"  [{i}] prompt: {ex['prompt']}")
            print(f"      reward_spec: {ex['reward_spec']}, max_turns: {ex['max_turns']}")
            ctx = ex["extra_info"]["context_text"]
            print(f"      context_text: {ctx[:120]}{'...' if len(ctx) > 120 else ''}")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, ds in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds.to_parquet(path)
    total = sum(len(ds) for ds in splits.values())
    print(f"\nWrote {len(splits)} splits ({total} total rows) to {args.output_dir}")


if __name__ == "__main__":
    main()

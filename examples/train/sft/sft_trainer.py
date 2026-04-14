"""
SFT trainer for alphaXiv page-relevance classification.

Fine-tunes a model to predict whether a PDF page is relevant to a user query,
given the paper's title, abstract, and the page text.

Dataset: alphaXiv/page-labels (binary label: 1=relevant, 0=irrelevant)

Usage:
    # export WANDB_API_KEY=<your_key>
    # export HF_TOKEN=<your_token>
    bash examples/train/sft/run_sft.sh
"""

import json
import os
import random
import shutil
from pathlib import Path

import ray
import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm

from ray.util.placement_group import placement_group

from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl.train.utils.utils import initialize_ray, validate_cfg, ResolvedPlacementGroup
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.tracking import Tracking

SYSTEM_PROMPT = (
    "You will be given a query, the paper's title and abstract, and the text of one page of a PDF. "
    "Rate the page's relevance to the query on a scale from 0 (not relevant) to 9 (highly relevant)."
)

EVAL_MD_PATH = Path(__file__).parents[3] / "EVAL.md"
THRESHOLDS = (4, 5, 6, 7)


def get_sft_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()

    num_gpus = int(os.environ.get("NUM_GPUS", "2"))

    cfg.trainer.policy.model.path = "Qwen/Qwen3.5-2B-Base"
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.policy.sequence_parallel_size = num_gpus
    cfg.trainer.logger = os.environ.get("LOGGER", "console")
    cfg.trainer.micro_train_batch_size_per_gpu = 64

    validate_cfg(cfg)
    return cfg


def build_chat_messages(example: dict) -> list[dict]:
    user_content = json.dumps(
        {
            "query": example["query"],
            "paperTitle": example["title"],
            "paperAbstract": example["abstract"],
            "pageNum": example["pageNumber"],
            "page": example["page"],
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def tokenize_sft_example(example: dict, tokenizer, max_length: int = 2048) -> dict:
    messages = build_chat_messages(example)
    label = max(0, min(9, example["label"] - 1))
    completion = str(label)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = prompt_text + completion

    full_tokens = tokenizer(full_text, add_special_tokens=False)

    if len(full_tokens["input_ids"]) > max_length:
        return {"input_ids": [], "attention_mask": [], "num_actions": 0, "label": label, "_valid": False}

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_tokens["input_ids"])
    full_len = len(full_tokens["input_ids"])
    num_actions = full_len - prompt_len

    if num_actions <= 0:
        return {"input_ids": [], "attention_mask": [], "num_actions": 0, "label": label, "_valid": False}

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "num_actions": num_actions,
        "label": label,
        "_valid": True,
    }


def collate_sft_batch(examples: list, tokenizer) -> TrainingInputBatch:
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)

    sequences = []
    attention_masks = []
    loss_masks = []

    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        sequences.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + ex["attention_mask"])
        action_pad = max_num_actions - ex["num_actions"]
        loss_masks.append([0] * action_pad + [1] * ex["num_actions"])

    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": max_num_actions}
    return batch


def run_validation(dispatch, val_tokenized, tokenizer, batch_size, num_eval_samples=10000, thresholds=THRESHOLDS):
    sample = random.sample(val_tokenized, min(num_eval_samples, len(val_tokenized)))

    candidate_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    num_total_batches = (len(sample) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(sample), batch_size), total=num_total_batches, desc="Validation"):
        batch_examples = sample[i : i + batch_size]
        if not batch_examples:
            continue

        batch = collate_sft_batch(batch_examples, tokenizer)
        batch.metadata["candidate_token_ids"] = candidate_ids
        num_batches += 1

        output = dispatch.forward("policy", batch)
        logprobs = output["output"]

        preds = logprobs.argmax(dim=-1).tolist()
        if isinstance(preds, int):
            preds = [preds]

        labels_t = torch.tensor([ex["label"] for ex in batch_examples], dtype=torch.long)
        correct_logp = logprobs[torch.arange(len(labels_t)), labels_t]
        total_loss += (-correct_logp.mean()).item()

        all_preds.extend(preds)
        all_labels.extend(ex["label"] for ex in batch_examples)

    avg_loss = total_loss / max(num_batches, 1)
    n = len(all_preds)
    avg_abs_dist = sum(abs(p - l) for p, l in zip(all_preds, all_labels)) / max(n, 1)

    logger.info(f"  avg_abs_distance={avg_abs_dist:.3f}")

    result = {"loss": avg_loss, "avg_abs_dist": avg_abs_dist}
    for t in thresholds:
        bin_preds = [int(p >= t) for p in all_preds]
        bin_labels = [int(l >= 6) for l in all_labels]
        tp = sum(p == 1 and g == 1 for p, g in zip(bin_preds, bin_labels))
        fp = sum(p == 1 and g == 0 for p, g in zip(bin_preds, bin_labels))
        fn = sum(p == 0 and g == 1 for p, g in zip(bin_preds, bin_labels))
        pct_pred_1 = sum(bin_preds) / n * 100 if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[f"t{t}/pct_pred_1"] = pct_pred_1
        result[f"t{t}/precision"] = precision
        result[f"t{t}/recall"] = recall
        result[f"t{t}/f1"] = f1

        logger.info(
            f"  threshold={t}  pred_1%={pct_pred_1:.1f}%  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}"
        )

    return result


def write_eval_md(eval_results, thresholds=THRESHOLDS):
    with open(EVAL_MD_PATH, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("**Dataset:** alphaXiv/page-labels (validation split)\n\n")
        f.write("### Overall\n\n")
        f.write("| Step | Val Loss | Avg Abs Dist |\n")
        f.write("|------|----------|--------------|\n")
        for step, m in eval_results:
            f.write(f"| {step} | {m['loss']:.4f} | {m['avg_abs_dist']:.3f} |\n")
        f.write("\n")
        for t in thresholds:
            f.write(f"### Threshold {t}\n\n")
            f.write("| Step | Val Loss | %Pred1 | Precision | Recall | F1 |\n")
            f.write("|------|----------|--------|-----------|--------|------|\n")
            for step, m in eval_results:
                f.write(
                    f"| {step} | {m['loss']:.4f} | {m[f't{t}/pct_pred_1']:.1f}% "
                    f"| {m[f't{t}/precision']:.3f} | {m[f't{t}/recall']:.3f} | {m[f't{t}/f1']:.3f} |\n"
                )
            f.write("\n")


def main():
    cfg = get_sft_config()
    initialize_ray(cfg)

    logger_backend = os.environ.get("LOGGER", "console")
    project_name = os.environ.get("WANDB_PROJECT", "alphaxiv-page-labels")
    run_name = os.environ.get("WANDB_RUN_NAME", "sft-qwen2.5-0.5b")

    tracker = Tracking(
        project_name=project_name,
        experiment_name=run_name,
        backends=logger_backend,
        config=cfg,
    )

    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    num_steps = int(os.environ.get("NUM_STEPS", "500"))
    eval_interval = int(os.environ.get("EVAL_INTERVAL", "50"))
    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "64"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-6"))

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset...")
    train_dataset = load_dataset("alphaXiv/page-labels", split="train")
    val_dataset = load_dataset("alphaXiv/page-labels", split="validation")

    num_workers = min(os.cpu_count() or 1, 8)

    logger.info(f"Tokenizing train dataset with {num_workers} workers...")
    train_mapped = train_dataset.map(
        lambda ex: tokenize_sft_example(ex, tokenizer, max_length),
        num_proc=num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    train_mapped = train_mapped.filter(lambda ex: ex["_valid"], num_proc=num_workers)
    tokenized_train = [
        {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"], "num_actions": ex["num_actions"], "label": ex["label"]}
        for ex in train_mapped
    ]
    random.shuffle(tokenized_train)
    logger.info(f"Train: kept {len(tokenized_train)}/{len(train_dataset)} examples")

    logger.info("Tokenizing validation dataset...")
    val_mapped = val_dataset.map(
        lambda ex: tokenize_sft_example(ex, tokenizer, max_length),
        num_proc=num_workers,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )
    val_mapped = val_mapped.filter(lambda ex: ex["_valid"], num_proc=num_workers)
    tokenized_val = [
        {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"], "num_actions": ex["num_actions"], "label": ex["label"]}
        for ex in val_mapped
    ]
    logger.info(f"Val: kept {len(tokenized_val)}/{len(val_dataset)} examples")

    logger.info("Initializing policy worker...")
    num_gpus = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
    pg = ResolvedPlacementGroup(raw_pg)

    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path, freeze_vision_encoder=True))

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
    dispatch.set_lr("policy", learning_rate)

    save_dir = Path(os.environ.get("SAVE_DIR", "checkpoints")).resolve()
    max_saved = 4
    saved_dirs: list[Path] = []

    eval_results = []
    logger.info(f"Starting SFT training for {num_steps} steps (eval every {eval_interval}), lr={learning_rate}...")
    logger.info(f"Tokenized train set size: {len(tokenized_train)}")

    for step in tqdm(range(num_steps)):
        start_idx = (step * batch_size) % len(tokenized_train)
        batch_examples = tokenized_train[start_idx : start_idx + batch_size]
        if len(batch_examples) < batch_size:
            batch_examples = tokenized_train[:batch_size]

        batch = collate_sft_batch(batch_examples, tokenizer)
        metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        grad_norm = dispatch.optim_step("policy")

        train_loss = metrics.get("final_loss", metrics.get("loss", 0.0))
        tracker.log({"train/loss": train_loss, "train/grad_norm": grad_norm}, step=step)

        if step % eval_interval == 0 or step == num_steps - 1:
            logger.info(f"Running validation at step {step}...")
            val_metrics = run_validation(dispatch, tokenized_val, tokenizer, eval_batch_size)
            log_dict = {"val/loss": val_metrics["loss"], "val/avg_abs_dist": val_metrics["avg_abs_dist"]}
            for t in THRESHOLDS:
                log_dict[f"val/t{t}/pct_pred_1"] = val_metrics[f"t{t}/pct_pred_1"]
                log_dict[f"val/t{t}/precision"] = val_metrics[f"t{t}/precision"]
                log_dict[f"val/t{t}/recall"] = val_metrics[f"t{t}/recall"]
                log_dict[f"val/t{t}/f1"] = val_metrics[f"t{t}/f1"]
            tracker.log(log_dict, step=step)
            eval_results.append((step, val_metrics))
            write_eval_md(eval_results)
            f1_summary = " ".join(f"f1@{t}={val_metrics[f't{t}/f1']:.3f}" for t in THRESHOLDS)
            logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, {f1_summary}")

            step_dir = save_dir / f"step_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving model to {step_dir}...")
            dispatch.save_hf_model("policy", str(step_dir), tokenizer)
            saved_dirs.append(step_dir)
            while len(saved_dirs) > max_saved:
                old = saved_dirs.pop(0)
                if old.exists():
                    logger.info(f"Removing old checkpoint {old}")
                    shutil.rmtree(old)


    write_eval_md(eval_results)
    logger.info(f"SFT training complete! Results written to {EVAL_MD_PATH}")
    tracker.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()

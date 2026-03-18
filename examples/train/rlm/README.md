# Training with RLMEnv

RLMEnv is a multi-turn environment where the model interacts with a Python REPL. Each episode has two text inputs:

- **Prompt** – direct instructions / tips the model sees in full in its context window (like a system prompt).
- **Context** – a large text stored externally in the REPL as the `context` variable. The model only sees a short preview and must run code to read more.

The model writes Python code (in ` ```python ... ``` ` blocks) to inspect `context`, optionally call `llm_query()`, and set `Final` when done. Reward is computed from `Final` vs `ground_truth`.

## Simplest way to start a training loop

### 1. Create an RLM dataset

Each row must have:

- **`prompt`**: list of messages — the direct instructions / tips the model sees in full.
- **`env_class`**: `"rlm"`.
- **`reward_spec`**: dict with **`ground_truth`** (required). Compared to `Final` (exact match → 1.0, numeric match → 1.0, substring → 0.5, else 0.0).
- Optionally **`max_turns`** (default 10).
- Optionally **`extra_info`**: dict that can include:
  - **`context_text`**: the external text stored in the REPL as the `context` variable. If not set, the `prompt` content is used as both the direct message and the REPL context.

Example: save as Parquet or JSON/JSONL with columns `prompt`, `env_class`, `reward_spec`, and optionally `max_turns`, `extra_info`.

You can generate a small dataset with:

```bash
uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $HOME/data/rlm
```

### 2. Launch training

**Recommended:** use the run script (same pattern as other examples like `gsm8k/run_gsm8k.sh`):

```bash
# Optional: prepare data first
uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $HOME/data/rlm

# Run training (override defaults via env: DATA_DIR, NUM_GPUS, LOGGER)
bash examples/train/rlm/run_rlm.sh
```

The script lives at **`examples/train/rlm/run_rlm.sh`** and is the right place to commit and tweak the training command (batch sizes, model path, epochs, etc.).

Required for RLM:

- **`environment.env_class=rlm`** – use RLMEnv.
- **`generator.step_wise_trajectories=true`** – multi-turn rollouts (RLM is multi-turn).
- **`generator.batched=false`** – step-wise generator does not support batched mode.
- **`generator.max_turns`** – e.g. `10` (must be > 1 for multi-turn).

Optional RLM env config (under `environment.skyrl_gym.rlm`):

- `metadata_prefix_length` (default 500) – chars of stdout shown in observations.
- `repl_timeout` (default 30.0) – timeout per REPL execution (seconds).
- `max_sub_calls_per_episode` (default 20) – max `llm_query()` calls per episode.
- `sub_model_url` – OpenAI-compatible endpoint for the frozen sub-LLM. If not set, `llm_query()` is not available.
- `sub_model_name` – model name for the sub-LLM endpoint.

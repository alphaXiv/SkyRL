set -x

# Eval-only multi-paper RLM run using OpenRouter as the inference backend,
# loading weights from a local FSDP checkpoint (e.g. global_step_130).
# Requires 8 GPUs (matching the world_size the checkpoint was saved with).
#
# 1. Set OPENROUTER_API_KEY in your environment.
# 2. Set RESUME_PATH to the global_step_N checkpoint directory.
# 3. Run: bash examples/train/rlm/train_scripts/run_multi_paper_rlm_openrouter_eval_from_ckpt.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
: "${UV_CACHE_DIR:=$PROJECT_ROOT/.uv-cache}"
: "${UV_PROJECT_ENVIRONMENT:=$PROJECT_ROOT/.venv}"
export UV_CACHE_DIR UV_PROJECT_ENVIRONMENT

: "${DATA_DIR:=$HOME/data/rlm-synthetic-multi}"

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
  echo "Data files missing — downloading alphaXiv/rlm-data-split from HuggingFace..."
  mkdir -p "$DATA_DIR"
  uv run --python 3.12 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='alphaXiv/rlm-data-split', repo_type='dataset', local_dir='$DATA_DIR')
"
  mv -f "$DATA_DIR/data/train-"*.parquet "$DATA_DIR/train.parquet"
  mv -f "$DATA_DIR/data/validation-"*.parquet "$DATA_DIR/validation.parquet"
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "Error: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

: "${RESUME_PATH:=/neer/artifacts/019d4d53-7c91-7052-b90b-5dedbeaf6580/019df632-1a39-76be-9674-287ef7c776e3/ckpts/rlm_ckpt/global_step_130}"

: "${OPENROUTER_MODEL:=anthropic/claude-sonnet-4.6}"

uv run --python 3.12 --extra skyrl-train -m examples.train.rlm.main_rlm_eval \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=multipaper_evidence_rlm \
  generator.step_wise_trajectories=true \
  generator.enable_child_agents=true \
  generator.max_turns=6 \
  generator.batched=false \
  generator.frozen_openrouter_model="$OPENROUTER_MODEL" \
  trainer.policy.model.path="alphaXiv/evidence-multi-rlm-sft-4b" \
  trainer.resume_mode="from_path" \
  trainer.resume_path="$RESUME_PATH" \
  trainer.placement.colocate_all=false \
  trainer.eval_interval=1 \
  trainer.eval_batch_size=1 \
  trainer.max_prompt_length=32768 \
  generator.max_input_length=32768 \
  generator.sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.chat_template_kwargs.enable_thinking=false \
  trainer.logger="['console']" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_multi_paper_openrouter_eval_ckpt" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.dump_eval_results=true \
  generator.rollout_dump_dir="$(pwd)/.neer/artifacts/rollouts" \
  "$@"

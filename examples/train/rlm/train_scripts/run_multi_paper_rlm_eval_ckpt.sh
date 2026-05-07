set -x

# Eval a local FSDP checkpoint using the full training stack (FSDP + vLLM).
# Loads weights from RESUME_PATH, syncs to vLLM, runs eval, then exits (epochs=0).
# Requires 8 GPUs.
#
# 1. Set RESUME_PATH to a global_step_N checkpoint directory (default is pre-filled).
# 2. Run: bash examples/train/rlm/train_scripts/run_multi_paper_rlm_eval_ckpt.sh

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

: "${NUM_ENGINES:=2}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=8}"
: "${INFERENCE_BACKEND:=vllm}"
: "${RESUME_PATH:=/root/weights/global_step_130}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

uv run --extra fsdp --python 3.12 -m examples.train.rlm.main_rlm \
  data.train_data="['$DATA_DIR/validation.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=multipaper_evidence_rlm \
  generator.step_wise_trajectories=true \
  generator.enable_child_agents=true \
  generator.train_child_trajectories=true \
  generator.max_turns=6 \
  generator.batched=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="alphaXiv/evidence-multi-rlm-sft-4b" \
  trainer.resume_mode="from_path" \
  trainer.resume_path="$RESUME_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.epochs=0 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.eval_batch_size=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  generator.max_input_length=32768 \
  generator.sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.inference_engine.enforce_eager=false \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=8 \
  trainer.logger="['console']" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_multi_paper_eval_ckpt" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/rlm_ckpt" \
  trainer.dump_eval_results=true \
  generator.rollout_dump_dir="$(pwd)/.neer/artifacts/rollouts" \
  "$@"

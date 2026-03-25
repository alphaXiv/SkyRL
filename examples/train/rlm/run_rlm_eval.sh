set -x

# RLM eval-only: generate rollouts and report metrics (no training).
#
# 1. Create data: uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/run_rlm_eval.sh

: "${DATA_DIR:=$HOME/data/rlm}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_PATH:=alphaXiv/rlm-sft-Qwen3.5-9B-v1}"

# Optional: fully replace the default RLM system prompt.
# Export CUSTOM_SYSTEM_PROMPT before running to override. Must be a complete
# system prompt (not just an addendum) — it replaces the default entirely.
# Uses heredoc-style YAML escaping so the prompt can contain backticks,
# curly braces, quotes, etc. literally.
_sq="'"
_YAML_PROMPT="${CUSTOM_SYSTEM_PROMPT//$_sq/$_sq$_sq}"
_YAML_PROMPT="${_sq}${_YAML_PROMPT}${_sq}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=15 \
  generator.batched=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.max_prompt_length=512 \
  generator.max_input_length=32768 \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_n_samples_per_prompt=1 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  trainer.dump_eval_results=true \
  trainer.export_path="$HOME/SkyRL/tmp/rlm-eval" \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_eval" \
  ${CUSTOM_SYSTEM_PROMPT:+environment.skyrl_gym.rlm.custom_system_prompt="$_YAML_PROMPT"} \
  "$@"

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
: "${MODEL_PATH:=Qwen/Qwen3.5-27B}"

# Sub-LLM for llm_query() inside the REPL. Set SUB_MODEL to enable.
# For OpenAI models (e.g. gpt-5-mini), just export OPENAI_API_KEY.
# For a local vLLM server, also set SUB_MODEL_URL.
: "${SUB_MODEL:=gpt-5.4-mini}"
: "${SUB_MODEL_URL:=}"

# Extra instructions appended to the default RLM system prompt.
# Uses a heredoc (with single-quoted delimiter to prevent any expansion)
# so the prompt can contain backticks, curly braces, quotes, etc. literally.
if [ -z "${SUPPLEMENTARY_SYSTEM_PROMPT:-}" ]; then
SUPPLEMENTARY_SYSTEM_PROMPT=$(cat <<'PROMPT_EOF'

<env_tips>
Strategy for long-context information retrieval:

1. Split the context into chunks (e.g., by paragraphs or fixed character windows with some overlap)
2. Write a prompt describing what to look for, then append it to each chunk to create a list of prompts
3. Call llm_query() many times with each prompt to scan chunks
4. Aggregate the relevant findings from the responses

Note: llm_query() uses gpt-5.4-mini which has a 400k token context window. Use this to inform your chunking strategy.

</env_tips>
PROMPT_EOF
)
fi

# YAML-escape the prompt for OmegaConf CLI: wrap in single quotes,
# doubling any internal single quotes (the only escape YAML single-quoted
# scalars need). This prevents YAML from interpreting colons, dashes, etc.
_sq="'"
_YAML_PROMPT="${SUPPLEMENTARY_SYSTEM_PROMPT//$_sq/$_sq$_sq}"
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
  ${SUB_MODEL:+environment.skyrl_gym.rlm.sub_model_name="$SUB_MODEL"} \
  ${SUB_MODEL_URL:+environment.skyrl_gym.rlm.sub_model_url="$SUB_MODEL_URL"} \
  ${SUPPLEMENTARY_SYSTEM_PROMPT:+environment.skyrl_gym.rlm.supplementary_system_prompt="$_YAML_PROMPT"} \
  "$@"

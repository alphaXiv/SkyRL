set -x

# RLM eval-only: generate rollouts and report metrics (no training).
#
# 1. Create data: uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/run_rlm_eval.sh

: "${DATA_DIR:=$HOME/data/rlm}"
: "${NUM_ENGINES:=2}"
: "${TP_SIZE:=4}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"
: "${MODEL_PATH:=Qwen/Qwen3.5-27B}"

# Sub-LLM for llm_query() inside the REPL. Set SUB_MODEL to enable.
# For OpenAI models (e.g. gpt-5-mini), just export OPENAI_API_KEY.
# For a local vLLM server, also set SUB_MODEL_URL.
: "${SUB_MODEL:=gpt-5-mini}"
: "${SUB_MODEL_URL:=}"

# Extra instructions appended to the default RLM system prompt.
: "${SUPPLEMENTARY_SYSTEM_PROMPT:=Be sure to submit a final answer to the user\'s question in the REPL. \
This should look like Final = \"my final answer\" or Final = my_answer_variable. \
Furthermore, you are only allowed to output one python code block per turn. \
Only the first code block will be executed and shown.
Keep any code logic short and concise.
You are encouraged to call lllm_query() and write python blocks **across multiple turns**.
In fact, it is impossible to answer the user question in a single turn.
DO NOT WRITE EOS UNTIL YOU HAVE SUBMITTED A FINAL ANSWER.
THIS MEANS THAT YOU MUST WRITE THE VALUE TO REPL.}"

uv run --extra fsdp -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=5 \
  generator.batched=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.max_prompt_length=512 \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.temperature=0.8 \
  generator.eval_n_samples_per_prompt=4 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.dump_eval_results=true \
  trainer.export_path="$HOME/SkyRL/tmp/rlm-eval" \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_eval" \
  ${SUB_MODEL:+environment.skyrl_gym.rlm.sub_model_name="$SUB_MODEL"} \
  ${SUB_MODEL_URL:+environment.skyrl_gym.rlm.sub_model_url="$SUB_MODEL_URL"} \
  ${SUPPLEMENTARY_SYSTEM_PROMPT:+environment.skyrl_gym.rlm.supplementary_system_prompt="$SUPPLEMENTARY_SYSTEM_PROMPT"} \
  "$@"

set -x

# Minimal RLM training run to validate that training with RLMEnv works.
# One step = 4 prompts × 2 samples = 8 rollouts, then one optimizer step.
#
# 1. Create tiny data: uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $DATA_DIR --n_train 8 --n_val 4
# 2. Run: bash examples/train/rlm/run_rlm.sh

: "${DATA_DIR:=$HOME/data/rlm}"
: "${NUM_GPUS:=1}"
: "${LOGGER:=console}"
: "${INFERENCE_BACKEND:=vllm}"

uv run --isolated --extra fsdp -v -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=5 \
  generator.batched=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=10 \
  trainer.eval_before_train=false \
  trainer.eval_interval=999 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.n_samples_per_prompt=2 \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_quick_validate" \
  trainer.resume_mode=null \
  trainer.log_path="$HOME/SkyRL/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/SkyRL/ckpts/rlm_ckpt" \
  "$@"

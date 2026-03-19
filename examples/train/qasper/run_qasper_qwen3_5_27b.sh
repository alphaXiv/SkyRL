set -x

# Colocated GRPO training+generation for Qwen3.5-27B-Instruct on QASPER.

# uv run examples/train/qasper/qasper_dataset.py --output_dir $HOME/data/qasper
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/qasper/run_qasper_qwen3_5_27b.sh

# You can override the default values with e.g.: `NUM_GPUS=8 bash examples/train/qasper/run_qasper_qwen3_5_27b.sh`.

export TRITON_PRINT_AUTOTUNING=1

: "${DATA_DIR:="$HOME/data/qasper"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"

TIS_IMP_RATIO_CAP=2.0
USE_TIS=true

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3.5-27B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_nodes=2 \
  trainer.placement.policy_num_nodes=2 \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  generator.inference_engine.num_engines=4 \
  generator.inference_engine.tensor_parallel_size=4 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  trainer.algorithm.use_tis=$USE_TIS \
  trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=4096 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=qasper \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="qasper-qwen3.5-27b" \
  trainer.run_name="qasper_qwen3.5_27B" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/qasper_qwen3.5_27B_ckpt" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@

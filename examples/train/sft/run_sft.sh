set -x

# SFT training for alphaXiv page-relevance classification.
#
# export WANDB_API_KEY=<your_key>
# export HF_TOKEN=<your_token>
# bash examples/train/sft/run_sft.sh

: "${LOGGER:=wandb}"
: "${NUM_GPUS:=4}"
: "${BATCH_SIZE:=256}"
: "${MICRO_BATCH_SIZE:=128}"
: "${NUM_STEPS:=16000}"
: "${MAX_LENGTH:=4096}"
: "${LEARNING_RATE:=2e-6}"
: "${EVAL_INTERVAL:=250}"
: "${EVAL_BATCH_SIZE:=1024}"
: "${NUM_EVAL_SAMPLES:=2048}"
: "${SAVE_INTERVAL:=1000}"
: "${SKIP_INITIAL_EVAL:=1}"
: "${WANDB_PROJECT:=alphaxiv-page-labels}"
: "${WANDB_RUN_NAME:=sft-qwen3.5-0.8b}"

export LOGGER NUM_GPUS BATCH_SIZE MICRO_BATCH_SIZE NUM_STEPS MAX_LENGTH LEARNING_RATE EVAL_INTERVAL EVAL_BATCH_SIZE
export NUM_EVAL_SAMPLES SAVE_INTERVAL SKIP_INITIAL_EVAL WANDB_PROJECT WANDB_RUN_NAME

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

uv run --directory "$REPO_ROOT" --isolated --extra fsdp --with flash-linear-attention \
	python examples/train/sft/sft_trainer.py "$@"

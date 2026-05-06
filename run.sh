set -x

# Colocated GRPO training+generation for Qwen3.5-0.8B on GSM8K.

uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
bash examples/train/models/run_qwen3.5_0.8b.sh
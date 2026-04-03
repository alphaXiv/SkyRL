# RLM Eval Speed Optimization Summary

## Baseline
- **Per-trajectory latency**: 31.7s
- **Total script wall time**: ~2m20s
- **Decode throughput**: ~37 tokens/sec on H200
- **Pass@1**: 0.0 (wrong answers)

## Result After Optimization
- **Per-trajectory latency**: 3.09s (**10x speedup**)
- **Total script wall time**: ~2m14s
- **Decode throughput**: ~125 tokens/sec
- **Pass@1**: 1.0 (correct answers)

---

## Optimizations Applied

### 1. Enable CUDA Graphs (`enforce_eager=false`)
**Impact: ~7x decode speedup (primary driver)**

`enforce_eager=True` (the default) runs every forward pass in eager mode — individual Python-level kernel launches with no batching. On an H200 with a 9B model, this produced only 37 tokens/sec (36ms/token) despite the GPU being capable of ~1000+ tokens/sec. Setting `enforce_eager=false` enables vLLM's PIECEWISE CUDA graph mode, which captures and replays the full decode graph in a single CUDA call. Per-step overhead dropped from 36ms to ~8ms.

**Config change in `run_rlm_eval.sh`:**
```
generator.inference_engine.enforce_eager=false
```

### 2. Fix Critical Bash Script Bug (all params after line 71 were silently dropped)
**Impact: fixes model quality + eliminates unnecessary thinking tokens (~3x fewer decode tokens)**

The script had bash comment lines (`# ...`) inserted inside a multi-line `uv run` command. In bash, a `#` comment terminates line continuation (`\`), so the entire `uv run` command ended at `language_model_only=true`. Everything after — including `enable_thinking=false`, `temperature=0.0`, `max_generate_length`, `speculative_config`, `gpu_memory_utilization=0.95`, the custom system prompt, and all remaining inference params — was silently ignored. The model was running with defaults and generating full thinking chains (`<think>...</think>`) every turn.

**Fix:** Removed comment lines from inside the `uv run` continuation. All params are now properly applied. With `enable_thinking=false` working correctly, the model emits only an empty `<think>\n\n</think>` (4 tokens) before answering, instead of 100–200 tokens of reasoning per turn.

### 3. Set `max_model_len=32768`
**Impact: reduces KV cache from 91GB → ~7GB; faster model init**

The model's native context window is 262,144 tokens. Without an explicit `max_model_len` override, vLLM allocates KV cache for the full 262K context, consuming 91GB of GPU memory. Since prompts are capped at 32,768 tokens, setting `max_model_len=32768` reduces KV cache allocation to match actual usage, saves ~84GB of HBM, and speeds up the memory profiling phase during model initialization.

**Config change in `run_rlm_eval.sh`:**
```
generator.inference_engine.engine_init_kwargs.max_model_len=32768
```

---

## Future Optimizations

### Speculative Decoding (ngram)
**BLOCKED: vLLM 0.18 v1 engine bug**

Attempted: `speculative_config={method: ngram, num_speculative_tokens: 5, prompt_lookup_max: 4}`. Engine crashes mid-generation with `AssertionError: num_required_blocks 17 < len(req_blocks) 18` in `single_type_kv_cache_manager.py`. This is a vLLM 0.18 bug with ngram spec decode in the v1 engine KV cache coordinator. The config reaches vLLM correctly (confirmed in logs) but fails at runtime. Skip until vLLM upgrade.

Additional constraint: `cudagraph_capture_sizes` must contain multiples of `(num_speculative_tokens + 1)`. With `num_spec_tokens=5`, minimum valid size is 6, not 1.

### MTP Speculative Decoding
**Estimated speedup: 1.5–3x on decode if model supports it**

The script already attempts to set `speculative_config={method: mtp, num_speculative_tokens: 1}`, but was not being applied due to the bash bug. Now that the bug is fixed, MTP can be re-enabled. Increasing `num_speculative_tokens` to 3–5 would give more benefit. Note: MTP requires the model to have multi-token prediction heads; verify `alphaXiv/rlm-sft-Qwen3.5-9B-v1` exposes them.

### Reduce CUDA Graph Compilation Time
**Impact: reduces total script wall time by ~50s; no inference latency benefit**

`cudagraph_capture_sizes=[1]` works via `engine_init_kwargs` and reduces graph compilation from ~60s to ~1s (confirmed: 52 sizes → 1 size). Engine init drops from ~19s to ~9s. However, this has no effect on per-trajectory inference latency. The total wall time benefit (~50s) is real but amortized over model loading (~60s) and Ray packaging overhead.

```bash
"generator.inference_engine.engine_init_kwargs.cudagraph_capture_sizes=[1]"
```

### Cache CUDA Graphs Across Runs
**Impact: eliminates CUDA graph compilation on subsequent runs**

vLLM supports a compilation cache directory (`compilation_config.cache_dir`). Setting this to a persistent path means CUDA graphs are compiled once and reused on subsequent runs, saving ~60s on every run after the first.

```bash
generator.inference_engine.engine_init_kwargs.compilation_config.cache_dir=/root/.vllm_cache
```

### Avoid FlashInfer GDN JIT on First Inference
**Potential: saves ~2s on first trajectory latency**

The Qwen3-Next model uses a FlashInfer GDN prefill kernel that JIT-compiles on first inference call ("first run may take a while"). This adds ~2s to the first trajectory. Setting `gdn_prefill_backend=triton` via `additional_config` avoids JIT but makes torch.compile 3x slower (23s vs 7s at init) and degrades decode throughput — net negative. 

The correct fix is to trigger a warmup prefill at engine startup so the JIT compiles before the timed eval starts. This requires a source change to the engine init or a separate warmup request before starting the eval loop.

### Reduce Ray File Packaging Overhead
**Impact: saves ~3–5s on Ray init**

Ray packages the entire working directory (315MB including worktrees) and uploads it to the GCS store on every run. `.rayignore` files are **not respected** by Ray — only `runtime_env={'excludes': [...]}` in the source code works. Large files to exclude: `data/qasper-*.json` (33–57MB each), `.git/objects/pack/` (38MB), `awscliv2.zip` (66MB). Requires modifying `prepare_runtime_environment()` in `skyrl/train/utils/utils.py`.

### Reduce Max Turns
**Impact: saves 1–3s if model terminates early anyway**

The dataset sets `max_turns=10`, but the model currently solves the task in 3 turns. Reducing `generator.max_turns` to 5 adds a safety margin while avoiding unnecessary overhead if the model ever fails to terminate early. This doesn't help when the model terminates naturally but caps worst-case latency.

### Keep vLLM Engine Warm (server mode)
**Impact: eliminates ~60–70s of model loading for repeated eval runs**

The dominant cost in total wall time is model loading (~60s) and CUDA graph compilation (~60s). For iterative eval workflows (running the script multiple times), a persistent vLLM server (via `enable_http_endpoint=true`) that stays alive between runs would reduce amortized per-run cost to just the inference time (~3s). This requires a process-management wrapper but no changes to SkyRL source code.

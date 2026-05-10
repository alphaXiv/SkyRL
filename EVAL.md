# SFT Training Samples

**Model:** `Qwen/Qwen3.5-4B` | **Dataset:** `alphaXiv/single-paper-sft-traces-entropy-collapse-ablation`

---

### Step 0 | loss=133.7018

```
```repl
s1 = search(context, "ImageNet", window=500)
s2 = search(context, "classification", window=500)
s3 = search(context, "accuracy", window=500)
```
```repl
s4 = search(context, "top-1", window=500)
s5 = search(context, "top-5", window=500)
s6 = search(context, "dataset", window=500)
```<|im_end|>
```

---

### Step 50 | loss=101.7541

```
```repl
s1 = search(context, "Chain of Thought", window=500)
s2 = search(context, "CoT", window=500)
s3 = search(context, "thinking traces", window=500)
s4 = search(context, "reasoning traces", window=500)
s5 = search(context, "thought", window=500)
```<|im_end|>
```

---

### Step 100 | loss=103.9281

```
```repl
s1 = search(context, "iterative self-improvement", window=500)
s2 = search(context, "self-improvement", window=500)
s3 = search(context, "generated reasoning traces", window=500)
```<|im_end|>
```

---

### Step 150 | loss=83.1563

```
```repl
s4 = search(context, "optimization algorithm", window=900)
s5 = search(context, "Proximal policy optimization", window=900)
s6 = search(context, "policy gradient", window=900)
```<|im_end|>
```

---

### Step 200 | loss=72.2538

```
```repl
s10 = search(context, "KV cache and activation quantization", window=1200)
s11 = search(context, "Potential reasons include differences in pretraining data", window=1200)
s12 = search(context, "more sensitive for KV cache and activation quantization", window=1200)
```

```repl
s10 = search(context, "KV cache and activation quantization", window=1200)
s11 = search(context, "Potential reasons include differences in pretraining data", window=1200)
s12 = search(context, "more sensitive for KV cache and activation quantization", window=1200)
```<|im_end|>
```

---

### Step 250 | loss=44.2782

```
```repl
FINAL_VAR(["Our model adapts CLIP to 3D CT volume and radiology text report pairs. It integrates a 3D encoder, previously used for generating CT volumes [17], as the image encoder, and CXR-Bert [41], a pre-trained model for chest X-ray reports, as the text encoder.", "Our model adapts CLIP to 3D CT volume and radiology text report pairs. It integrates a 3D encoder, previously used for generating CT volumes [17], as the image encoder, and CXR-Bert [41], a pre-trained model for chest X-ray reports, as the text encoder. Similar to the 2D CLIP approach, our method extracts low-dimensional latent embeddings from input volumes and corresponding sections of radiological reports."])
```<|im_end|>
```

---

### Step 300 | loss=36.6927

```
```repl
FINAL_VAR([p3, p5])
```<|im_end|>
```


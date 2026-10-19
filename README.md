# Fast linear-attention kernels on an HPC cluster, installed at the user level

How to make Hugging Face `transformers` use the fast Triton kernels of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(fla) for hybrid linear-attention models such as **Qwen3.5 / Qwen3.8**, when you run inside a read-only container on a cluster where you have no root:
no image rebuild, no compilation, nothing written outside one directory you own. On an NVIDIA H200 with Qwen3.8-27B this made forward passes
**2-4x faster** with numerically equivalent outputs.

## Why this was needed

Qwen3.5-architecture models (`model_type: qwen3_5`, e.g. `Qwen/Qwen3.8-27B`) mix full-attention layers with linear-attention layers (gated delta rule;
Qwen3.8-27B: 48 of 64 layers). vLLM serves them with its own kernels, but research code that needs **forward hooks on the residual stream** (probes,
activation patching, ablations) has to run the model in Hugging Face `transformers`. There, every job printed:

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because `flash-linear-attention` is not installed.
`causal_conv1d_fn` is falling back to its reference PyTorch implementation because `causal_conv1d` is not installed.
```

The reference path is correct but slow: 6.6 s per forward pass of a 16k-token prompt on an H200. Interpretability work repeats such passes thousands of
times (one activation-patching scan over 64 layers is 64 passes per stimulus pair).

The convenient environment on many clusters is a prebuilt container (here the official `vllm/vllm-openai` image, vLLM 0.31.0, transformers 5.17.0,
torch 2.13.0+cu130, triton 3.7.1, run with Singularity/Apptainer). It has torch and Triton, but not fla, and the image is read-only.

## The method

fla is pure Python + Triton kernels: **it needs no CUDA compilation**, only a Triton that matches the container's torch, which the container already has.
So it can live outside the image:

1. **Install with the container's own pip into a plain directory** (`pip install --target DIR`), on a filesystem the compute nodes can see.
   Using the container's pip guarantees the same Python version and ABI as at run time.
2. **`--no-deps`**: torch, triton, einops and transformers already come with the container. Letting pip resolve them could place a second torch/triton
   inside `DIR`, which would then shadow the container's builds at import time.
3. **Put `DIR` on the container's `PYTHONPATH` only for the jobs that want it**, by exporting `SINGULARITYENV_PYTHONPATH` / `APPTAINERENV_PYTHONPATH`
   (the runtime turns `SINGULARITYENV_<NAME>` into `<NAME>` inside the container). Default jobs are unchanged, so earlier results stay reproducible.
4. **Nothing to change in your code.** In transformers 5.x the reference function is declared as
   `@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`, so the model switches to fla automatically when `import fla` works.
   The check is that the `chunk_gated_delta_rule ... falling back` warning disappears.

```bash
# 1. install (login node with internet; the container's pip is used)
SIF=/path/to/vllm-openai.sif TARGET=/shared/fs/me/pylib bash scripts/install_fla_user_level.sh
#    -> Successfully installed fla-core-0.5.2 flash-linear-attention-0.5.2
#    -> fla 0.5.2 | torch 2.13.0+cu130 | triton 3.7.1      (a "roll back to CPU" warning on a GPU-less login node is harmless)

# 2. run a job with the kernels on (FLA=1) or off (default)
sbatch --export=ALL,FLA=1,SIF=/path/to/vllm-openai.sif,PYLIB=/shared/fs/me/pylib examples/sbatch_snippet.sh your_script.py --args
```

Add your site's partition/QOS/constraint flags to `sbatch`, and adjust the bind mount (`-B /scratch`) in `examples/sbatch_snippet.sh`.

## Verification: is it used, is it faster, is it the same?

`scripts/verify_kernels.py` loads the model in transformers, records which fallback warnings appear, times forward passes on fixed token sequences
(token ids 1000, 1001, ... cycled; a benchmark input, not data) after one warm-up pass, and saves the last-token logits. Run it once with `FLA=1` and once
without, then compare with `scripts/compare_logits.py`.

Measured on one NVIDIA H200, Qwen/Qwen3.8-27B in bf16, transformers 5.17.0, torch 2.13.0+cu130, fla 0.5.2 (median of 3 passes after 1 warm-up):

| sequence length | reference (s / forward) | fla (s / forward) | speed-up | top-1 token | top-10 overlap | KL(ref ‖ fla) | max abs logit diff |
|---|---|---|---|---|---|---|---|
| 1,024 | 0.708 | 0.332 | 2.1x | same | 10/10 | 2.2e-4 | 0.072 |
| 4,096 | 1.895 | 0.472 | 4.0x | same | 10/10 | 6.5e-5 | 0.063 |
| 16,384 | 6.607 | 1.931 | 3.4x | same | 10/10 | 2.9e-7 | 0.344 |

Fallback warnings: reference run 2 (`causal_conv1d_fn`, `chunk_gated_delta_rule`), fla run 1 (`causal_conv1d_fn` only).
Raw output: `results/benchmark_h200_qwen3.8-27b.txt`.

In a real workload (14-17k-token chat prompts, a yes/no log-odds readout) the steady-state time per pass went from ~5.9-7.0 s to ~1.6-2.0 s, and the
readout computed from bf16 logits was bit-identical on 5 of 6 prompts and differed by one bf16 rounding step (0.125) on the sixth.

## Things to know

- **Triton compiles per new input shape.** The first pass at a new sequence length took 25-36 s with fla (vs 2-21 s warm-up for the reference). Workloads with
  many distinct lengths pay this repeatedly at the start; batch or bucket lengths if that matters.
- **The numbers are not bit-identical** to the reference path (different algorithm in bf16). Do not mix the two inside one comparison; record which one each run
  used (the job log prints `[env] fla kernels ON/OFF`).
- **`causal_conv1d` is still the reference version.** It needs a CUDA build (nvcc and torch headers matching the container), which this method avoids.
  The speed-ups above are with fla alone.
- **Pin versions.** `install_fla_user_level.sh` installs `flash-linear-attention==0.5.2` and `fla-core==0.5.2` by default (`FLA_VERSION` to change).
  A newer fla may need a newer Triton than the container has.
- Prefix caching (running a shared prompt prefix once and continuing from the cache) was not exact for this model in bf16 with the reference path
  (log-odds differed by up to 0.085 from a full pass); check this yourself before relying on cached continuations for measurements.
- Tested with one container image, one model and one GPU type. Other Qwen3.5-family models (e.g. MoE variants) use the same layer code and should behave
  the same; not tested here.

## Files

```
scripts/install_fla_user_level.sh   install fla into a directory with the container's pip (no root, no compilation)
examples/sbatch_snippet.sh          SLURM job: FLA=1 puts the directory on the container's PYTHONPATH
scripts/verify_kernels.py           which implementation is used, seconds per forward pass, last-token logits (.npz)
scripts/compare_logits.py           compare two verify runs: logit differences, top-k agreement, KL, speed-up
results/benchmark_h200_qwen3.8-27b.txt   the measured run reported above
```

## License

MIT (see `LICENSE`). flash-linear-attention, transformers, vLLM and the model weights keep their own licenses.

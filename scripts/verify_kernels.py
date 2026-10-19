#!/usr/bin/env python3
"""Check which linear-attention implementation transformers uses, time forward passes, and save the last-token logits for comparison.

Run it twice in the same container on the same GPU type, once with the fla directory on PYTHONPATH and once without, then compare the two
.npz files with compare_logits.py.

The input is a fixed, deterministic token sequence (token ids 1000, 1001, ... cycled), used only to time the model and to compare the two
implementations on identical input. It is a benchmark input, not data.

Usage: python3 verify_kernels.py --model-dir /path/to/hf/model --out run_fla.npz [--lengths 1024,4096,16384] [--repeats 3]
"""
import argparse
import importlib.util
import json
import logging
import platform
import time

import numpy as np
import torch


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="1024,4096,16384")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    import transformers
    from transformers import AutoConfig
    h = Collect()
    for name in ("transformers", ""):
        logging.getLogger(name).addHandler(h)
    cfg = AutoConfig.from_pretrained(a.model_dir)
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    load_s = time.time() - t0
    info = {"fla_importable": importlib.util.find_spec("fla") is not None,
            "fla_version": __import__("fla").__version__ if importlib.util.find_spec("fla") else None,
            "causal_conv1d_importable": importlib.util.find_spec("causal_conv1d") is not None,
            "transformers": transformers.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(), "model_type": cfg.model_type, "load_s": round(load_s, 1)}
    logits, timing = {}, {}
    for n in [int(x) for x in a.lengths.split(",")]:
        ids = torch.tensor([[1000 + (i % 5000) for i in range(n)]], device="cuda")
        ts = []
        for r in range(a.repeats + 1):                 # first pass = warm-up (Triton JIT compiles on first use)
            torch.cuda.synchronize()
            t1 = time.time()
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
            ts.append(time.time() - t1)
        timing[n] = {"warmup_s": round(ts[0], 3), "median_s": round(float(np.median(ts[1:])), 3)}
        logits[f"len{n}"] = out.logits[0, -1].float().cpu().numpy()
        print(n, timing[n], flush=True)
    fallback = [m for m in h.msgs if "falling back" in m]
    info.update({"timing_s_per_forward": timing, "fallback_warnings": fallback})
    np.savez(a.out, **logits, info=json.dumps(info))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

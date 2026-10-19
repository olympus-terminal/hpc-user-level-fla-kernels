#!/usr/bin/env python3
"""Compare the last-token logits of two verify_kernels.py runs (e.g. fla vs reference) on identical input.

Reports per sequence length: max |diff|, mean |diff|, top-1 agreement, top-10 overlap, KL(p_a || p_b) of the softmax distributions, and the speed-up.
Usage: python3 compare_logits.py run_reference.npz run_fla.npz
"""
import json
import sys

import numpy as np


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    ia, ib = json.loads(str(a["info"])), json.loads(str(b["info"]))
    print("A:", sys.argv[1], "fla" if ia["fla_importable"] else "reference", "| B:", sys.argv[2], "fla" if ib["fla_importable"] else "reference")
    for k in sorted([k for k in a.files if k.startswith("len")], key=lambda s: int(s[3:])):
        x, y = a[k], b[k]
        pa, pb = softmax(x.astype(np.float64)), softmax(y.astype(np.float64))
        kl = float(np.sum(pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))))
        t10a, t10b = set(np.argsort(-x)[:10]), set(np.argsort(-y)[:10])
        n = k[3:]
        ta, tb = ia["timing_s_per_forward"][n]["median_s"], ib["timing_s_per_forward"][n]["median_s"]
        print(f"{k:>9}: max|d| {np.abs(x - y).max():.4f}  mean|d| {np.abs(x - y).mean():.4f}  top1 {'same' if x.argmax() == y.argmax() else 'DIFF'}  "
              f"top10 overlap {len(t10a & t10b)}/10  KL {kl:.2e}  time A {ta:.3f}s B {tb:.3f}s  speed-up A/B {ta / tb:.1f}x")


if __name__ == "__main__":
    main()

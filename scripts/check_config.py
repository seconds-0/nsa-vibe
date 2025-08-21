#!/usr/bin/env python3
import json
import sys
from omegaconf import OmegaConf


def reads(S: int, l: int, d: int, n: int, l_sel: int, w: int) -> int:
    num_cmp = 0 if S < l else (S - l) // d + 1
    return num_cmp + n * l_sel + min(w, S)


def main():
    if len(sys.argv) < 2:
        print("Usage: check_config.py <config.yaml>")
        sys.exit(1)
    cfg = OmegaConf.load(sys.argv[1])
    l = int(cfg.nsa.l)
    d = int(cfg.nsa.d)
    l_sel = int(cfg.nsa.l_sel)
    n_sel = int(cfg.nsa.n_sel)
    w = int(cfg.nsa.w)

    if l % d != 0 or l_sel % d != 0:
        print("ERROR: require d|l and d|l_sel (M0)")
        sys.exit(2)

    out = {
        "l": l, "d": d, "l_sel": l_sel, "n_sel": n_sel, "w": w,
        "reads@S": {S: reads(S, l, d, n_sel, l_sel, w) for S in [0, 64, 128, 256, 512, 1024, 4096]},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


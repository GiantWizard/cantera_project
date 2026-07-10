# build training grid: run_reactor over 100 log-spaced tau values

import time
import numpy as np
import pandas as pd

from reactor_model import run_reactor

TAU_MIN = 3e-4   # s, safely above the blowout limit (~2.6e-4 s)
TAU_MAX = 0.5    # s
N_POINTS = 100

if __name__ == "__main__":
    taus = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_POINTS)

    t0 = time.perf_counter()
    x_co = []
    for i, tau in enumerate(taus):
        val = run_reactor(tau)
        x_co.append(val)
        print(f"[{i+1:3d}/{N_POINTS}] tau={tau:.6e}  X_CO={val:.6e}")
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame({"tau": taus, "X_CO": x_co})
    df.to_csv("grid_data.csv", index=False)

    with open("grid_timing.txt", "w") as f:
        f.write(f"Grid generation: {N_POINTS} points, wall-clock time = {elapsed:.4f} s\n")
        f.write(f"Average per Cantera run = {elapsed / N_POINTS * 1000:.3f} ms\n")

    print(f"\nSaved grid_data.csv ({N_POINTS} rows)")
    print(f"Total wall-clock time: {elapsed:.4f} s ({elapsed/N_POINTS*1000:.3f} ms/run)")

# Cantera Surrogate Optimization

Simulates a small combustion chamber in Cantera, fits a fast model that copies the simulator's behavior, and uses that fast model to search for the input that maximizes carbon monoxide output. The found answer then gets checked against the real simulation to see how close the fast model actually got.

The idea shows up a lot in process systems engineering: real simulations are often too slow to search directly, so you run them a limited number of times, train something quick on those runs, and search the quick version instead. This project builds a toy-scale version of that pattern rather than reproducing any specific paper.

## The setup

A well-stirred reactor burns methane and air at a stoichiometric ratio, tracked with GRI-Mech 3.0. The input is residence time (`tau`, in seconds), how long gas sits in the chamber before flowing out. The output is the steady-state fraction of CO in the exhaust. Below roughly `tau = 2.6e-4 s` the flame blows out entirely. Just above that, CO peaks near 3% because combustion is fast but incomplete, then falls to around 0.9% by `tau = 0.5 s` as the extra time lets CO oxidize further into CO2. The optimum sits right at the edge of that blowout boundary, which makes it a harder target for a surrogate than a smooth interior maximum would be.

## Pipeline

`reactor_model.py` holds the ground-truth simulator. `generate_data.py` runs it across 100 log-spaced points from `tau = 3e-4` to `0.5` seconds and saves the grid. `fit_surrogates.py` fits two surrogates on that grid, a CatBoost model tuned with Optuna and a Gaussian Process with an RBF kernel, both in log-tau space since the response is smoother there. `optimize_and_verify.py` searches each surrogate for its predicted optimum with another 200 Optuna trials, then runs the real simulator at that point to see how far off the surrogate actually was.

## Results

The GP tracked the real simulator almost exactly, landing within 0.001% of the true optimum in both the original run and a from-scratch reproduction. CatBoost was consistently less precise at that boundary, off by 1.4% to 2.1% depending on the run, since it predicts in flat steps rather than a smooth curve and this particular optimum sits right against a sharp edge. Full numbers, both runs, and the reproducibility check are in [RESULTS.md](RESULTS.md).

Worth being honest about: at this scale, one input and a single Cantera run costing about 100ms, the surrogate doesn't actually save meaningful time over just running the real simulator directly. The exercise demonstrates the method rather than solving a real bottleneck. It would start paying off with more decision variables or a stiffer, slower mechanism, which is the natural next step.

## Running it

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python generate_data.py
python fit_surrogates.py
python optimize_and_verify.py
```

Cantera needs Python 3.12 or 3.13; it doesn't publish wheels for 3.14 yet. GRI-Mech 3.0 ships inside the `cantera` package, so no separate download is needed.

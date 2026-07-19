# Cantera Substitute Optimization

This project simulates a combustion chamber in Cantera, then fits a model copying the simulation's behavior. It then uses the model to search for the input that maximizes carbon monoxide output, which is finally checked against the real simulation.

## Why I built this

This idea shows up a lot in process systems engineering: because real simulations are often too slow to search directly, you can train a fast, substitute model on some limited amount of real simulation runs and use the model to search. This model is a toy-scale version of this idea.

## The setup

The simulation tracks a stoiciometric methane-air burn inside a perfectly-stirred reactor (PSR) with the GRI-Mech 3.0 mechanism, mapping residence time (`tau`) to the steady-state fraction of CO in the exhaust.

- Below `tau = 2.6e-4 s`L The velocity is too high and the flame blows out completely.
- Just above the boundary: Ignition holds but combustion is cut short and CO peaks near 3% due to an incomplete reaction
- Approaching `tau = 0.5 s`: The extra time allows the intermediate CO to fully oxidize into stable CO_2, dropping the fraction to ~0.9%.

Since the global optimum sits on the edge of the boundary (rather than a smooth interior curve) it serves as a rigorous stress test for surrogate ML models. 

## Pipeline

- `reactor_model.py` holds the simulator engine
- `generate_data.py` runs `reactor_model.py` across 100 log-spaced points from `tau = 3e-4` to `0.5` seconds and saves the grid.
- `fit_surrogates.py` fits two surrogates on that grid, a CatBoost model tuned with Optuna and a Gaussian Process with an RBF kernel, both in log-tau space (for a smoother response). 
- `optimize_and_verify.py` searches each substitute for its predicted optimum with another 200 Optuna trials, then runs the real simulator at that point to see how far off the substitute actually was.

## Results

The GP tracked the real simulator almost exactly, landing within 0.001% of the true optimum in both the original run and a from-scratch reproduction. CatBoost was consistently less precise at that boundary, off by 1.4% to 2.1% depending on the run, since it predicts in steps rather than a curve and this particular optimum sits right against an edge. Numbers and runs are in [RESULTS.md](RESULTS.md).

At this scale, one input and a single Cantera run cost about 100ms, thus the substitutes don't actually save meaningful time over just running the actual simulator directly. The exercise demonstrates the methodology rather than solving a existing bottleneck. It would start paying off only with more decision variables or a slower mechanism.

## Running it

```
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python generate_data.py
python fit_surrogates.py
python optimize_and_verify.py
```

Note: Cantera needs Python 3.12 or 3.13; it doesn't publish wheels for 3.14 yet. GRI-Mech 3.0 ships inside the `cantera` package, so no separate download is needed.

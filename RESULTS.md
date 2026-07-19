# Cantera Surrogate Optimization: Results

This project is a small-scale demonstration of a core methodology used widely in systems engineering. It demonstrates surrogate-based optimization, where instead of querying an expensive simulation every time you want to test a new thing you run it a few times to train a fast ML model that closely mimics the simulator's behavior. You can then rapidly search for the optimal input with the fast model, and when a promising answer shows up you verify it against the large, slow simulation to ensure the surrogate is accurate. This project serves as a proof of concept of this idea.

## Step 1. Build the base model

The thing being simulated is a small combustion chamber. The chamber is a well-stirred reactor where methane and air flow in continuously at a stoichiometric ratioand hot exhaust flows out the other side. It's tracked with GRI-Mech 3.0, a detailed kinetic mechanism that models roughly fifty chemical species across hundreds of individual reactions. 

In Cantera, this is built from three standard pieces: an `IdealGasReactor` for the chamber, a `MassFlowController` that pushed fresh gas in a set rate, and a `PressureController` that lets exhaust out fast enough hold pressure steady. This is implemented in `reactor_model.py`.
 - Input: residence time `tau`, in seconds
 - Output: The steady-state fraction of carbon monoxide in the exhaust, `X_CO`

## Step 2. Mapping Reactor Behavior & Optimization Bounds

Before building any models, the behavorial landscape of the reactor has to be mapped. A coarse initial scan across a broad window (`tau = 0.001` to `5` seconds) showed a flat trend, but zooming in closer showed a much more volatile behavior profile:
- Blowout (Approaching `tau = 2.6e-4 s`): The gas's velocity outpaces the flame's speed, flushing reactants out of the chamber before they can ignite, extinguishing the flame.
- Peak: The flame anchors itself but the rapid throughput cuts the reaction chain short. CO is trapped at its peat near 3% of the exhaust
- Decay (Approaching `tau = 0.5 s`): The gas lingers in the hot reactor, providing time for the intermediate CO to oxidize into stable CO_2, dropping the CO fraction down to ~0.9%

The search range was set between `3e-4` and `0.5` seconds to isolate the 3.3x performace swing while keeping the flame sustained. Because the true optimum sits right on the edge of a blowout rather than an interior slope, the reactor serves as a stress test for surrogate models that over-smooth the boundary resulting in an over- or underestimate of the maximum.

## Step 3. Grid generation

- 100 points are evenly spaced on a log scale from 3e-4 to 0.5 seconds
- The model took 10.35s to run (from the original 12.05s run, though not due to any code change)
- Saved to `grid_data.csv`

At this small scale, one Cantera run costs about 100-120 ms, so calling it an "expensive simulator" is more about demonstrating the process than any actual issue. The real payoff grows with the input space: Optuna needed about 200 trials to search each surrogate below, and running 400 Cantera calls directly woul still finish in under a minute at this scale, but this gap would widen exponentially with more decision vairables and a bigger chemical mechanism.

## Step 4. Fit the surrogates

Both surrogates were fit in `log10(tau)` since the CO response is smoother on a log scale (common practice when tau spans several orders of magnitude). The data was split 80/20 into train and test sets, with 15% of the training set being held as a validation set for CatBoost's early stopping.

| Surrogate | Train R² | Test R² |
|---|---|---|
| CatBoost (tuned with Optuna over 25 trials, reusing the same tuning setup from the `chemical_property_predictor` project) | 0.9981 - 0.9991 | 0.9953 - 0.9977 |
| Gaussian Process (RBF + WhiteKernel, via `sklearn`) | 1.0000 | 1.0000 |

CatBoost builds an ensemble of decision trees, where each new tree corrects the mistakes left over from the trees before it. A Gaussian Process instead fits one smooth, probabilistic curve directly to the data, using a kernel function that describes how strongly nearby points should be correlated. Optuna automatically searches over a model's settings, its hyperparameters, to find the best-performing combination instead of someone picking them by hand.

Both surrogates fit almost perfectly here, which is expected for a smooth one-dimensional function with only 100 points. The real test is the optimize-and-verify step below, not the fit. The CatBoost range in the table reflects two runs of the pipeline: the original build and a rerun (verification). Optuna also isnt seeded, so the scores differ a little -- just noise. The GP is deterministic given the same training data.

CatBoost's tuned hyperparameters were `iterations=530, learning_rate=0.0747, depth=3, l2_leaf_reg=3.20, rsm=0.904` in the original run, versus `iterations=707, learning_rate=0.0150, depth=3, l2_leaf_reg=0.464, rsm=0.810` in the verification run, different points in hyperparameter space that landed on similarly good fits.

The GP's fitted kernel was identical in both runs: `15.4**2 * RBF(length_scale=1.35) + WhiteKernel(noise_level=1e-12)`.

## Step 5. Optimization and verification 

Optuna was used  with 200 trials to search for the tau that each surrogate predicts will produce the highest X_CO. Then `run_reactor()`, the actual Cantera model, was run at that exact tau to see how close each surrogate's guess came to reality.

**Original run:**

| Surrogate | tau* found | Surrogate-predicted X_CO | Real (verified) X_CO | Gap |
|---|---|---|---|---|
| CatBoost | 3.5006e-4 s | 0.029118 | 0.029545 | +4.27e-4 (**+1.45%** of real) |
| GP | 3.0011e-4 s | 0.030441 | 0.030441 | +4.15e-7 (**+0.001%** of real) |

**Verification re-run (new virtual environment, dependencies reinstalled):**

| Surrogate | tau* found | Surrogate-predicted X_CO | Real (verified) X_CO | Gap |
|---|---|---|---|---|
| CatBoost | 3.3024e-4 s | 0.029256 | 0.029876 | +6.20e-4 (**+2.08%** of real) |
| GP | 3.0008e-4 s | 0.030442 | 0.030442 | +4.16e-7 (**+0.001%** of real) |

For reference, the best point that existed in the 100-point training grid was tau=3.000e-4 s, X_CO=0.030444, which is essentially the same answer the GP arrives at in both runs.

Both surrogates correctly located the optimum at the edge of the search range closer to blowout, which makes sense because CO peaks right after ignition before there's time to oxidize it away. The GP tracked the real simulator almost exactly in both runs (as it interpolates smoothly and the training grid sampled points against the boundary), while CatBoost was a little less prcise because it predicts in constant steps rather than a curve, underestimating the optimum by 1.45% originally and 2.08% in the rerun. The exact size of the error varies with Optuna's unseeded search, but the GP reliably beat CatBoost at pinning down the optimum.

## Reproducibility

Results were rerun once, in a freshly rebuilt virtual environment, to confirm they held up outside the original setup. The grid data (X_CO values) came back bit-identical, since the underlying physics are deterministic. CatBoost's hyperparameters and optimization gap shifted a little between the two runs, as expected since Optuna, again, is not seeded (see the tables above), while the GP's result was essentially identical both times.

## Setup friction

- Cantera installed cleanly with `pip install cantera` inside a **Python 3.12 virtual environment**. My machine's default `python3` was version 3.14, which is too new for Cantera: Cantera 3.2.0 hasn't publish wheels for it yet, so a 3.12 environment was created at `cantera_project/venv` for this project. That was the only issue, and once on 3.12, installing Cantera and loading GRI-Mech 3.0 worked immediately with no errors.

## Scope and possible extensions

This is a small toy problem, with one input and one output, and at this scale, again, a single Cantera run is fast enough, about 100 ms. It is mostly about demonstrating the method than solving a bottleneck. Of course, the next step is naturally a higher-dimensional version where a single simulation costs enough and the search space is big enough that fitting and optimizng a surrogate becomes practical instead of demonstration on a trivial case such as this one.

## Files in this folder

- `reactor_model.py`: the `run_reactor(tau)` function, the base simulator
- `generate_data.py`: generates the training grid, produces `grid_data.csv` and `grid_timing.txt`
- `grid_curve.png`: a sanity-check plot of the grid data
- `fit_surrogates.py`: fits both surrogates, produces `catboost_surrogate.cbm`, `gp_surrogate.joblib`, `surrogate_metrics.txt`
- `optimize_and_verify.py`: produces `optimization_results.txt`
- `requirements.txt`: the dependency list used to build the virtual environment
- `venv/`: the Python 3.12 virtual environment (Cantera, CatBoost, Optuna, scikit-learn, etc.), rebuilt from scratch during verification
- `_old_run_backup/`: output files from the original run, kept around for the run-to-run comparison documented above

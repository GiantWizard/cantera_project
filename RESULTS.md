# Cantera Surrogate Optimization: Results

This project is a small, toy-scale version of an idea that comes up a lot in Professor Ilias Mitrai's research area: instead of running an expensive computer simulation every time you want to test a new input, you run it a modest number of times, use those results to train a fast machine learning model that copies its behavior, and then search for the best input using that fast copy instead of the real thing. Once a promising answer turns up, you go back and check it against the real, slow simulation to make sure the fast model wasn't lying to you. This project does not copy any specific paper of hers. Her published work with Prodromos Daoutidis, including the review "Accelerating process control and optimization via machine learning" (arXiv:2412.18529), is about using machine learning to speed up and configure the optimization algorithms themselves, choosing and tuning the search method for large, structured optimization problems, rather than building a stand-in for one physics simulator the way this project does. The connection is thematic: both are about using data-driven shortcuts to speed up expensive optimization work, not a case of this project reproducing one of her specific methods.

## The physical model

The thing being simulated is a small combustion chamber, essentially a tiny furnace that fuel and air flow into continuously while hot exhaust flows out the other side, known as a well-stirred reactor. Methane and air go in in exactly the ratio needed for complete burning (called stoichiometric), and the actual chemistry happening inside is tracked with GRI-Mech 3.0, a widely used, detailed model of the roughly fifty chemical species and hundreds of reactions involved when methane burns. In Cantera, the simulation library used here, this setup is built from a few standard pieces: an `IdealGasReactor` for the chamber itself, a `MassFlowController` that pushes fresh gas in at a set rate, and a `PressureController` that lets exhaust out fast enough to hold the pressure steady. This is Cantera's standard recipe for modeling this kind of reactor.

- **Input**: residence time `tau`, in seconds, meaning how long on average a parcel of gas sits inside the chamber before it flows back out
- **Output**: the steady-state fraction of carbon monoxide in the exhaust, `X_CO`

The relationship between CO and tau is not simple. If tau is too short, below roughly 2.6e-4 s, gas leaves before it can properly ignite and the flame blows out entirely. Just above that point, CO peaks around 0.030 (three percent of the exhaust) because combustion is happening fast but has not gone all the way to completion. As tau grows, there is more time for that CO to get oxidized further into CO2, and it falls to around 0.009 by tau = 0.5 s. The search range used here, tau between 3e-4 and 0.5 seconds, sits just inside the region where the flame stays lit and captures this entire rise-then-fall shape, a real roughly 3.3x swing in CO level confirmed by plotting the curve (`grid_curve.png`) before building anything on top of it.

Because the best tau for maximizing CO sits right at the edge of the search range, next to where the flame would blow out, this is a harder test than it first looks. A surrogate model that smooths over that steep, unstable edge could easily over- or under-guess how much CO is actually achievable there.

## Grid generation (Step 3)

- 100 points, evenly spaced on a log scale across tau from 3e-4 to 0.5 seconds
- **Total wall-clock time: 10.35 s (103.5 ms per Cantera run)** in the from-scratch verification re-run described below. The original run took 12.05 s (120.5 ms per run). That roughly 15% difference came from how busy the computer happened to be at the time, not from any code change: the actual X_CO values produced were identical bit for bit between the two runs, since the underlying physics is deterministic.
- Saved to `grid_data.csv`

This timing is the number that actually matters for explaining why anyone would bother with a surrogate here at all. At this small scale, one input, a simple sweep, a moderate chemical mechanism, a single Cantera run only costs about 100 to 120 milliseconds, so calling it an "expensive simulator" is more about demonstrating the method than describing real pain. The real payoff of surrogates shows up once the input space grows. Optuna needed 200 trials to search each surrogate below, and running 400 real Cantera calls directly instead would still finish in under a minute at this scale. That gap widens fast once there are more decision variables to search over, or a bigger, stiffer chemical mechanism to solve.

## Surrogates (Step 4)

Both surrogates were fit in `log10(tau)` space rather than raw tau, since the CO response looks much smoother on a log scale, which is standard practice when tau spans several orders of magnitude like it does here. The data was split 80/20 into train and test sets, and 15% of the training set was carved out separately as a validation set used only for CatBoost's early stopping, meaning training halts once the model stops improving so it does not overfit.

| Surrogate | Train R² | Test R² |
|---|---|---|
| CatBoost (tuned with Optuna over 25 trials, reusing the same tuning setup from the `chemical_property_predictor` project) | 0.9981 - 0.9991 | 0.9953 - 0.9977 |
| Gaussian Process (RBF + WhiteKernel, via `sklearn`) | 1.0000 | 1.0000 |

CatBoost builds an ensemble of decision trees, where each new tree corrects the mistakes left over from the trees before it. A Gaussian Process instead fits one smooth, probabilistic curve directly to the data, using a "kernel" function that describes how strongly nearby points should be correlated. Optuna is a library that automatically searches over a model's settings, its hyperparameters, to find the best-performing combination instead of someone picking them by hand.

Both surrogates fit almost perfectly here, which is expected since this is a smooth one-dimensional function with only 100 points to fit, not a hard problem. The real test is not how well they fit the training data, it is the optimize-and-verify step below. The CatBoost range in the table reflects two separate runs of the whole pipeline: the original build and a from-scratch verification re-run in a freshly created virtual environment. Optuna's search is not seeded with a fixed random state, so it lands on different hyperparameters each time it runs, and the R² scores between the two runs differ by about 0.002 to 0.003, which is ordinary run-to-run noise rather than a problem. The GP, on the other hand, is deterministic given the same training data, so its result is identical across both runs.

CatBoost's tuned hyperparameters were `iterations=530, learning_rate=0.0747, depth=3, l2_leaf_reg=3.20, rsm=0.904` in the original run, versus `iterations=707, learning_rate=0.0150, depth=3, l2_leaf_reg=0.464, rsm=0.810` in the fresh verification run, two different points in hyperparameter space that both landed on similarly good fits.

The GP's fitted kernel was identical in both runs: `15.4**2 * RBF(length_scale=1.35) + WhiteKernel(noise_level=1e-12)`.

## Optimization and verification (Step 5)

Optuna was used again, this time with 200 trials, to search for the tau that each surrogate predicts will produce the highest X_CO. Then `run_reactor()`, the real Cantera model, was actually run at that exact tau to see how close each surrogate's guess came to reality.

**Original run:**

| Surrogate | tau* found | Surrogate-predicted X_CO | Real (verified) X_CO | Gap |
|---|---|---|---|---|
| CatBoost | 3.5006e-4 s | 0.029118 | 0.029545 | +4.27e-4 (**+1.45%** of real) |
| GP | 3.0011e-4 s | 0.030441 | 0.030441 | +4.15e-7 (**+0.001%** of real) |

**Fresh verification re-run (new virtual environment, dependencies reinstalled, same code):**

| Surrogate | tau* found | Surrogate-predicted X_CO | Real (verified) X_CO | Gap |
|---|---|---|---|---|
| CatBoost | 3.3024e-4 s | 0.029256 | 0.029876 | +6.20e-4 (**+2.08%** of real) |
| GP | 3.0008e-4 s | 0.030442 | 0.030442 | +4.16e-7 (**+0.001%** of real) |

For reference, the single best point that actually existed in the 100-point training grid was tau=3.000e-4 s, X_CO=0.030444, essentially the same answer the GP arrives at in both runs.

Both surrogates correctly figured out that the true optimum sits right at the edge of the search range closest to blowout, which makes physical sense, since CO peaks right after ignition, before there has been time to oxidize it away. The GP tracked the real simulator almost exactly in both runs, a 0.001% gap and the same tau to three significant figures, because it interpolates smoothly and the training grid already sampled points right up against that boundary. CatBoost predicts in flat, piecewise-constant steps rather than a smooth curve, so it is naturally less precise right at a sharp edge like this one: it landed slightly inside the range in both runs, underestimating the true optimum by 1.45% in the original run and 2.08% in the fresh re-run. The exact size of that gap shifts with Optuna's unseeded search, but the pattern held up across both independent runs, with the GP surrogate reliably beating CatBoost at pinning down this particular boundary optimum.

## Reproducibility

Results were re-run once, in a freshly rebuilt virtual environment, to confirm they held up outside the original setup. The grid data (X_CO values) came back bit-identical, since the underlying physics is deterministic. CatBoost's hyperparameters and optimization gap shifted a little between the two runs, as expected since Optuna is not seeded (see the tables above), while the GP's result was essentially identical both times.

## Setup friction

- Cantera installed cleanly with `pip install cantera` inside a **Python 3.12 virtual environment**. This machine's default `python3` was version 3.14, which is too new: Cantera 3.2.0 does not publish wheels for it yet, so a separate 3.12 environment was created at `cantera_project/venv` just for this project. That was the only real snag. Once on 3.12, installing Cantera and loading GRI-Mech 3.0 worked immediately with no errors, confirmed again during the from-scratch re-run with no new friction the second time around.
- Cantera 3.2 prints `DeprecationWarning`s about the older `Reactor.thermo` attribute name and the `clone` default. Harmless, the code still runs correctly, but worth flagging for anyone extending this later since Cantera's interface is partway through a transition right now.
- No separate GRI-Mech download was needed. `gri30.yaml` ships bundled inside the `cantera` pip package already.
- Finding a search range that actually showed interesting behavior took one extra pass. A first, coarse scan from 0.001 to 5 seconds looked roughly monotonic and only mildly interesting. A finer scan closer to the blowout limit revealed the sharp extinction edge and the peak-then-decay shape that ended up in the final grid.

## Scope and possible extensions

This is a small, one-dimensional toy problem: one input (residence time), one output (CO fraction), and at this scale a single Cantera run is fast enough, about 100 ms, that calling it an "expensive simulator" is more about demonstrating the method than describing a real bottleneck. The surrogate-versus-real tradeoff does not fully pay off yet at this size. A natural next step is a higher-dimensional version, for example treating inlet temperature, equivalence ratio, and residence time as three decision variables at once, or adding a second reactor with heat integration, where a single real simulation costs enough and the search space is big enough that fitting and optimizing a surrogate becomes a genuine practical win instead of a demonstration on an easy case.

## Files in this folder

- `reactor_model.py`: the `run_reactor(tau)` function, the ground-truth simulator
- `generate_data.py`: Step 3, generates the training grid, produces `grid_data.csv` and `grid_timing.txt`
- `grid_curve.png`: a sanity-check plot of the grid data
- `fit_surrogates.py`: Step 4, fits both surrogates, produces `catboost_surrogate.cbm`, `gp_surrogate.joblib`, `surrogate_metrics.txt`
- `optimize_and_verify.py`: Step 5, produces `optimization_results.txt`
- `requirements.txt`: the dependency list used to build the virtual environment
- `venv/`: the Python 3.12 virtual environment (Cantera, CatBoost, Optuna, scikit-learn, etc.), rebuilt from scratch during verification
- `_old_run_backup/`: output files from the original run, kept around for the run-to-run comparison documented above

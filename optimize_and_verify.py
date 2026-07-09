"""Step 5: use Optuna to find the tau that maximizes X_CO according to each
surrogate, then run the real Cantera simulation at that tau and compare."""

import numpy as np
import optuna
import joblib
import pandas as pd
from catboost import CatBoostRegressor

from reactor_model import run_reactor
from generate_data import TAU_MIN, TAU_MAX

optuna.logging.set_verbosity(optuna.logging.WARNING)

cat_model = CatBoostRegressor()
cat_model.load_model("catboost_surrogate.cbm")

gp_bundle = joblib.load("gp_surrogate.joblib")
gp_model, gp_scaler = gp_bundle["model"], gp_bundle["scaler"]

LOG_TAU_MIN, LOG_TAU_MAX = np.log10(TAU_MIN), np.log10(TAU_MAX)


def optimize_surrogate(predict_fn, name):
    def objective(trial):
        log_tau = trial.suggest_float("log_tau", LOG_TAU_MIN, LOG_TAU_MAX)
        return predict_fn(log_tau)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)
    best_log_tau = study.best_params["log_tau"]
    best_tau = 10 ** best_log_tau
    predicted = study.best_value
    print(f"[{name}] optimal tau = {best_tau:.6e} s  (predicted X_CO = {predicted:.6e})")
    return best_tau, predicted


def cat_predict(log_tau):
    return float(cat_model.predict(np.array([[log_tau]]))[0])


def gp_predict(log_tau):
    return float(gp_model.predict(gp_scaler.transform(np.array([[log_tau]])))[0])


cat_tau, cat_pred = optimize_surrogate(cat_predict, "CatBoost")
gp_tau, gp_pred = optimize_surrogate(gp_predict, "GP")

# This is the whole point of the project: check each surrogate's guess
# against the real, slow simulator instead of trusting its prediction.
cat_real = run_reactor(cat_tau)
gp_real = run_reactor(gp_tau)

cat_gap = cat_real - cat_pred
gp_gap = gp_real - gp_pred

# best point actually present in the training grid, for reference
grid = pd.read_csv("grid_data.csv")
true_best_row = grid.loc[grid["X_CO"].idxmax()]

report = f"""
=== Optimization results ===
CatBoost:  tau*={cat_tau:.6e} s   predicted X_CO={cat_pred:.6e}   REAL X_CO={cat_real:.6e}   gap={cat_gap:+.6e} ({100*cat_gap/cat_real:+.3f}% of real)
GP:        tau*={gp_tau:.6e} s   predicted X_CO={gp_pred:.6e}   REAL X_CO={gp_real:.6e}   gap={gp_gap:+.6e} ({100*gp_gap/gp_real:+.3f}% of real)

Grid reference (best of the 100 training points):
  tau={true_best_row['tau']:.6e} s   X_CO={true_best_row['X_CO']:.6e}
"""
print(report)

with open("optimization_results.txt", "w") as f:
    f.write(report)

# fit CatBoost (Optuna-tuned) and GP surrogates on the grid data

import joblib
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("grid_data.csv")

# tau spans several orders of magnitude; the response is smooth on a log scale
X = np.log10(df[["tau"]].values)
y = df["X_CO"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.3, 1.0),
        "loss_function": "RMSE",
        "early_stopping_rounds": 50,
        "verbose": False,
    }
    m = CatBoostRegressor(**params)
    m.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return np.sqrt(mean_squared_error(y_val, m.predict(X_val)))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

best_params = study.best_params
best_params.update({"loss_function": "RMSE", "early_stopping_rounds": 50, "verbose": False})
cat_model = CatBoostRegressor(**best_params)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
cat_model.save_model("catboost_surrogate.cbm")

cat_train_r2 = r2_score(y_train, cat_model.predict(X_train))
cat_test_r2 = r2_score(y_test, cat_model.predict(X_test))

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-6, (1e-12, 1e-1))
gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=42)
gp_model.fit(X_train_s, y_train)

gp_train_r2 = r2_score(y_train, gp_model.predict(X_train_s))
gp_test_r2 = r2_score(y_test, gp_model.predict(X_test_s))

print("=== CatBoost ===")
print(f"Best Optuna params: {study.best_params}")
print(f"Train R2: {cat_train_r2:.5f}   Test R2: {cat_test_r2:.5f}")

print("\n=== Gaussian Process ===")
print(f"Fitted kernel: {gp_model.kernel_}")
print(f"Train R2: {gp_train_r2:.5f}   Test R2: {gp_test_r2:.5f}")

with open("surrogate_metrics.txt", "w") as f:
    f.write("CatBoost surrogate\n")
    f.write(f"  best params: {study.best_params}\n")
    f.write(f"  train R2: {cat_train_r2:.5f}\n")
    f.write(f"  test R2:  {cat_test_r2:.5f}\n\n")
    f.write("Gaussian Process surrogate\n")
    f.write(f"  fitted kernel: {gp_model.kernel_}\n")
    f.write(f"  train R2: {gp_train_r2:.5f}\n")
    f.write(f"  test R2:  {gp_test_r2:.5f}\n")

joblib.dump({"model": gp_model, "scaler": scaler}, "gp_surrogate.joblib")

print("\nSaved catboost_surrogate.cbm, gp_surrogate.joblib, surrogate_metrics.txt")

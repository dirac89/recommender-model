import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "verbose": 0
    }
    model = CatBoostRegressor(**params)
    return -cross_val_score(model, X, y, cv=3, scoring="neg_root_mean_squared_error").mean()

def tune_hyperparameters(X, y, n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    return study.best_params
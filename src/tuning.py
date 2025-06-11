# tuning.py

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def tune_models(X_train, y_train, cv_folds=5):
    """
    Perform GridSearchCV for multiple regressors and return the best estimator.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    pipelines = {
        'ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]),
        'lasso': Pipeline([('scaler', StandardScaler()), ('lasso', Lasso())]),
        'svr':   Pipeline([('scaler', StandardScaler()), ('svr', SVR())]),
        'rf':    Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor(random_state=42))]),
        'mlp':   Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(max_iter=500, random_state=42))])
    }

    param_grids = {
        'ridge': {'ridge__alpha': [0.1, 1.0, 10.0]},
        'lasso': {'lasso__alpha': [0.01, 0.1, 1.0]},
        'svr':   {'svr__C': [0.1, 1, 10], 'svr__kernel': ['rbf', 'linear']},
        'rf':    {'rf__n_estimators': [50, 100], 'rf__max_depth': [5, 10]},
        'mlp':   {'mlp__hidden_layer_sizes': [(50,), (100,)], 'mlp__alpha': [0.0001, 0.001]}
    }

    best_score = -np.inf
    best_model = None
    best_name = None

    for name, pipeline in pipelines.items():
        grid = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=kf,
            scoring='r2',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_name = name

    print(f"Best model: {best_name} with CV R²={best_score:.3f}")
    return best_model

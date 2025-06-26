import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

param_grid = {
    "alpha": np.logspace(-2, 1, 10),
    "l1_ratio": np.linspace(0.1, 0.9, 9)
}

model = ElasticNet(max_iter=10000)
grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X, y)

print("En iyi alpha:", grid.best_params_['alpha'])
print("En iyi l1_ratio:", grid.best_params_['l1_ratio'])
print("En iyi R2 skoru:", grid.best_score_)

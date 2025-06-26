import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV


def plot_predictions(y_test, y_pred, model_name):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.title(f"{model_name} - Gerçek vs Tahmin")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()

def xgboost_example():
    from xgboost import XGBRegressor
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n[XGBoosting Boosting Regressor]")
    print("Ortalama Karesel Hata (MSE):", mean_squared_error(y_test, y_pred))
    print("R2 Skoru:", r2_score(y_test, y_pred))
    plot_predictions(y_test, y_pred, "XGBoosting")


def gradient_boosting_example():
    from sklearn.ensemble import GradientBoostingRegressor
    X,y=fetch_california_housing(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print("\n[Gradient Boosting Regressor]")
    print("Ortalama Karesel Hata (MSE):", mean_squared_error(y_test, y_pred))
    print("R2 Skoru:", r2_score(y_test, y_pred))
    plot_predictions(y_test, y_pred,"GradientBoosting")


def lightgbm_example():
    from lightgbm import LGBMRegressor
    # ✅ as_frame=True ile DataFrame olarak al
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    # ✅ train/test bölme işlemini doğrudan DataFrame ve Series ile yap
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Modeli oluştur ve eğit
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Tahmin ve skor
    y_pred = model.predict(X_test)
    print("\n[LightGBM]")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 :", r2_score(y_test, y_pred))

    # ✅ Grafik çiz
    plot_predictions(y_test, y_pred, "LightGBM")

def catboost_example():
    from catboost import CatBoostRegressor
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = CatBoostRegressor(iterations=100, depth=5,learning_rate=0.1, verbose=False, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n[CatBoost Regressor]")
    print("Ortalama Karesel Hata (MSE):", mean_squared_error(y_test, y_pred))
    print("R2 Skoru:", r2_score(y_test, y_pred))
    plot_predictions(y_test, y_pred,"CatBoost")

def quantile_example():
    from sklearn.ensemble import GradientBoostingRegressor
    X,y=fetch_california_housing(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model = GradientBoostingRegressor(loss='quantile',alpha=0.9,n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred_q90 = model.predict(X_test)
    print("\n[Quantile Regressor (0.9)]")
    print("Ortalama Karesel Hata (MSE):", mean_squared_error(y_test, y_pred_q90))
    print("R2 Skoru:", r2_score(y_test, y_pred_q90))
    plot_predictions(y_test, y_pred_q90,"Quantile Regressor")

def timeseries_boosting_example():
    from lightgbm import LGBMRegressor
    X,y=fetch_california_housing(return_X_y=True)
    tscv=TimeSeriesSplit(n_splits=5)
    scores=[]
    print("\n[TimeSeries Boosting + LightGBM]")
    for fold ,(train_ix, test_ix) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        scores.append((mse, r2))

        print(f"Fold {fold+1} - MSE: {mse:.4f}, R2: {r2:.4f}")
        plot_predictions(y_test, y_pred, "TimeSeries Boosting + LightGBM")

def xgboost_gridsearch_example():
    from xgboost import XGBRegressor
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print("\n[XGBoost GridSearchCV]")
    print("Best score:", grid.best_score_)
    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    plot_predictions(y_test, y_pred, "XGBoost GridSearchCV")


if __name__ == "__main__":
    gradient_boosting_example()
    xgboost_example()
    lightgbm_example()
    catboost_example()
    quantile_example()
    timeseries_boosting_example()
    xgboost_gridsearch_example()

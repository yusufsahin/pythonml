import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn
print("scikit-learn:", sklearn.__version__)

# 1. Veri setini yükle
data = fetch_openml(name="house_prices", as_frame=True)
df = data.frame.copy()
y = df["SalePrice"].astype(float)
X = df.drop("SalePrice", axis=1)

# 2. Kolon türlerini ayır
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 3. Eksik veri oranı yüksek olanları göster
print("Eksik veri oranı yüksek olanlar:\n", X.isnull().mean().sort_values(ascending=False).head(10))
# 4. Preprocessing pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])
# 5. Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Pipeline: preprocessing + model
base_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(random_state=42))
])

# 7. GridSearchCV ile optimizasyon
param_grid = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, 15, 20],
    "rf__min_samples_split": [2, 5]
}
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="r2",
    verbose=2
)

print("\n GridSearch başlatılıyor...")
grid_search.fit(X_train, y_train)

# 8. En iyi sonuçlar
print("\n✅ En iyi parametreler:", grid_search.best_params_)
print("⭐ En iyi cross-val R²:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# 9. Test seti değerlendirmesi
y_pred = best_model.predict(X_test)
print("\n📊 Test seti performansı:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))
# 10. SHAP ile açıklanabilirlik
print("\n🔎 SHAP hesaplanıyor...")
rf_model = best_model.named_steps["rf"]
X_transformed = best_model.named_steps["preprocess"].transform(X_test)
feature_names = best_model.named_steps["preprocess"].get_feature_names_out()

explainer = shap.Explainer(rf_model)
shap_values = explainer(X_transformed)
# SHAP summary plot
shap.plots.beeswarm(shap_values)

# Opsiyonel: İlk örnek için detaylı açıklama
# shap.plots.waterfall(shap_values[0])

# 11. Modeli kaydet
joblib.dump(best_model, "rf_ames_optimized_model.pkl")
print("\n💾 Model kaydedildi: rf_ames_optimized_model.pkl")
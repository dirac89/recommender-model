import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generar dataset artificial con una feature dominante
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "feature_dominant": np.random.normal(loc=100, scale=1, size=n),          # Altamente correlacionada
    "feature_secondary": np.random.normal(loc=0, scale=10, size=n),          # Poco señal
    "feature_noise": np.random.uniform(0, 1, size=n),                         # Ruido
})
df["target"] = df["feature_dominant"] * 0.9 + np.random.normal(0, 0.5, n)

X = df.drop(columns="target")
y = df["target"]

# 2. Entrenar modelo sin regularización
model_no_reg = CatBoostRegressor(verbose=0, l2_leaf_reg=1.0)
model_no_reg.fit(X, y)

# 3. Entrenar modelo con regularización fuerte
model_reg = CatBoostRegressor(verbose=0, l2_leaf_reg=25.0)
model_reg.fit(X, y)

# 4. SHAP
explainer_no_reg = shap.Explainer(model_no_reg)
shap_values_no_reg = explainer_no_reg(X)

explainer_reg = shap.Explainer(model_reg)
shap_values_reg = explainer_reg(X)

# 5. Importancia promedio por feature
importance_no_reg = pd.DataFrame({
    "feature": X.columns,
    "importance": np.abs(shap_values_no_reg.values).mean(axis=0),
    "model": "No regularization"
})

importance_reg = pd.DataFrame({
    "feature": X.columns,
    "importance": np.abs(shap_values_reg.values).mean(axis=0),
    "model": "L2 regularization (25)"
})

importance_all = pd.concat([importance_no_reg, importance_reg])

# 6. Visualización
plt.figure(figsize=(10, 5))
sns.barplot(data=importance_all, x="feature", y="importance", hue="model")
plt.title("SHAP Feature Importance Comparison")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.show()
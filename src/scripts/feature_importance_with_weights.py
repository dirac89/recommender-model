import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generar dataset artificial
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "feature_dominant": np.random.normal(loc=100, scale=1, size=n),
    "feature_secondary": np.random.normal(loc=0, scale=10, size=n),
    "feature_noise": np.random.uniform(0, 1, size=n),
})
df["target"] = df["feature_dominant"] * 0.9 + np.random.normal(0, 0.5, n)

X = df.drop(columns="target")
y = df["target"]

# 2. Definir sample_weight: penalizamos valores extremos de feature_dominant
mean_dom = X["feature_dominant"].mean()
std_dom = X["feature_dominant"].std()
sample_weight = np.exp(-((X["feature_dominant"] - mean_dom) / std_dom) ** 2)

# 3. Entrenar modelo baseline (sin pesos)
model_base = CatBoostRegressor(verbose=0, l2_leaf_reg=1.0)
model_base.fit(X, y)

# 4. Entrenar modelo con sample_weight
model_weighted = CatBoostRegressor(verbose=0, l2_leaf_reg=1.0)
model_weighted.fit(X, y, sample_weight=sample_weight)

# 5. SHAP
explainer_base = shap.Explainer(model_base)
shap_base = explainer_base(X)

explainer_weighted = shap.Explainer(model_weighted)
shap_weighted = explainer_weighted(X)

# 6. Importancia SHAP promedio
importance_base = pd.DataFrame({
    "feature": X.columns,
    "importance": np.abs(shap_base.values).mean(axis=0),
    "model": "Baseline"
})

importance_weighted = pd.DataFrame({
    "feature": X.columns,
    "importance": np.abs(shap_weighted.values).mean(axis=0),
    "model": "Weighted (sample_weight)"
})

importance_all = pd.concat([importance_base, importance_weighted])

# 7. Visualización
plt.figure(figsize=(10, 5))
sns.barplot(data=importance_all, x="feature", y="importance", hue="model")
plt.title("SHAP Feature Importance Comparison (sample_weight)")
plt.ylabel("Mean |SHAP value|")
plt.tight_layout()
plt.show()
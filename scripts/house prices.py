# Project 4: House Price Prediction Pipeline with Feature Importance Analysis

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv("data/house_prices_dataset.csv")

# 2. Data Cleaning & Feature Engineering
df_clean = df.drop(columns=["zipcode"]).copy()
df_clean["house_age"] = 2026 - df_clean["year_built"]
df_clean["is_renovated"] = (df_clean["year_renovated"] > 0).astype(int)

# Separate features (X) and target (y)
X = df_clean.drop(columns=["price"])
y = df_clean["price"]

# 3. Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Feature Importance Analysis (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=True
)

# Plot Feature Importance Chart
plt.figure(figsize=(10, 6))
importances.plot(kind="barh", color="#2b5c8f")
plt.title(
    "Feature Importances for House Price Prediction",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()
plt.savefig("screenshots/feature_importance.png", dpi=300)
plt.close()

# 6. Model Training & Benchmark Comparison
models = {
    "Lasso Regression": (Lasso(alpha=1000.0, max_iter=10000), True),
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=1.0), True),
    "Gradient Boosting": (
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        False,
    ),
    "Random Forest": (
        RandomForestRegressor(n_estimators=100, random_state=42),
        False,
    ),
}

eval_results = []
for name, (model, is_scaled) in models.items():
    if is_scaled:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)  # Fits the model before predicting
        preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    eval_results.append(
        {"Model": name, "MAE ($)": mae, "RMSE ($)": rmse, "R2 Score": r2}
    )

print("\n=================== MODEL PERFORMANCE LEADERBOARD ===================")
print(pd.DataFrame(eval_results).to_string(index=False))

# 7. Actual vs Predicted Plot
lasso_model = Lasso(alpha=1000.0, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train)
best_preds = lasso_model.predict(X_test_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    y_test / 1000,
    best_preds / 1000,
    alpha=0.7,
    color="#d95f02",
    edgecolors="k",
)
plt.plot(
    [y_test.min() / 1000, y_test.max() / 1000],
    [y_test.min() / 1000, y_test.max() / 1000],
    "k--",
    lw=2,
)
plt.title(
    "Actual vs Predicted House Prices (Lasso Regression)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Actual Price ($1,000s)", fontsize=12)
plt.ylabel("Predicted Price ($1,000s)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("screenshots/actual_vs_predicted.png", dpi=300)
plt.close()

# 8. Model Serialization
with open("models/house_price_model.pkl", "wb") as f:
    pickle.dump(lasso_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(
    "\n[SUCCESS] Execution complete! Visualisations and model pkl files saved successfully."
)
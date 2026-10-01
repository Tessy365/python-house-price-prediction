# House Price Prediction Pipeline (Python & Machine Learning)
## Executive Summary
This project builds an end-to-end Machine Learning pipeline to predict residential housing prices using real estate structural and spatial features. The dataset comprises 1,000 property records. The pipeline handles data preprocessing, feature engineering, scaling, model evaluation across 5 algorithms, and model serialization. The top-performing **Lasso Regression** model achieves an **$R^2$ score of 0.9966** with a Mean Absolute Error of **$15,363.22**.

---

## Machine Learning Workflow

```text
1. Data Preprocessing ──> 2. Feature Engineering ──> 3. Feature Scaling
                                                               │
6. Serialization <── 5. Feature Importance <── 4. Model Benchmarking
```
---
##  Repository Structure
```text

python-house-price-prediction/
├── data/                       # Real estate dataset (.csv)
├── models/                     # Serialized model & scaler (.pkl)
├── notebooks/                  # Interactive Jupyter Notebook (.ipynb)
├── report/                     # Written findings report (.pdf)
├── screenshots/                # Model evaluation & feature importance plots
├── scripts/                    # Python pipeline script (.py)
└── README.md                   # Complete project documentation
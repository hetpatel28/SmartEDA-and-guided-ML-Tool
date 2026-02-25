import pandas as pd


# =====================================================
# 1️⃣ DETECT PROBLEM TYPE
# =====================================================

def detect_problem_type(df, target):

    unique_values = df[target].nunique()

    if df[target].dtype in ["object", "category"]:
        return "Classification"

    # numeric but low unique values → likely classification
    if unique_values < 15:
        return "Classification"

    return "Regression"


# =====================================================
# 2️⃣ SUGGEST MODELS
# =====================================================

def suggest_models(problem_type):

    if problem_type == "Classification":
        return [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "Support Vector Machine",
            "K-Nearest Neighbors"
        ]
    else:
        return [
            "Linear Regression",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "Ridge Regression",
            "Lasso Regression"
        ]


# =====================================================
# 3️⃣ SUGGEST METRICS
# =====================================================

def suggest_metrics(problem_type):

    if problem_type == "Classification":
        return [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    else:
        return [
            "MAE",
            "MSE",
            "RMSE",
            "R² Score"
        ]
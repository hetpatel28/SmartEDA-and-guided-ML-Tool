import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


# =====================================================
# 1️⃣ Feature Importance (Tree Models)
# =====================================================

def get_feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        return df

    return None


# =====================================================
# 2️⃣ Coefficient Importance (Linear Models)
# =====================================================

def get_coefficients(model, feature_names):

    if hasattr(model, "coef_"):
        coefs = model.coef_

        if len(coefs.shape) > 1:
            coefs = coefs[0]

        df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        return df

    return None


# =====================================================
# 3️⃣ SHAP Explanation (Tree Models Only)
# =====================================================

def generate_shap_summary(model, X_sample):

    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)

    return fig


# =====================================================
# 4️⃣ Human-Readable Insights
# =====================================================

def generate_insights(importance_df, problem_type):

    top_features = importance_df.head(3)["Feature"].tolist()

    if problem_type == "Regression":
        return (
            f"The model predictions are mainly influenced by: "
            f"{', '.join(top_features)}. "
            f"Changes in these features strongly impact the predicted values."
        )
    else:
        return (
            f"The classification outcome is strongly driven by: "
            f"{', '.join(top_features)}. "
            f"These features contribute most to class separation."
        )
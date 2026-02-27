import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1️⃣ Evaluate Models
# =====================================================

def evaluate_models(problem_type, trained_models, predictions, y_test):

    results = []

    for name, preds in predictions.items():

        if problem_type == "Classification":

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4)
            })

        else:

            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)

            results.append({
                "Model": name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2 Score": round(r2, 4)
            })

    return pd.DataFrame(results)


# =====================================================
# 2️⃣ Cross Validation
# =====================================================

def perform_cross_validation(problem_type, model, X, y):

    if problem_type == "Classification":
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    else:
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    return round(scores.mean(), 4)


# =====================================================
# 3️⃣ Confusion Matrix Plot
# =====================================================

def plot_confusion_matrix(y_test, preds):

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return fig


# =====================================================
# 4️⃣ ROC Curve
# =====================================================

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt


def plot_roc_curve(model, X_test, y_test):

    n_classes = len(np.unique(y_test))

    # ===============================
    # Binary Classification
    # ===============================
    if n_classes == 2:

        y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        return plt.gcf()

    # ===============================
    # Multiclass Classification
    # ===============================
    else:

        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test)

        plt.figure()

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {classes[i]} AUC={roc_auc:.2f}")

        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curve (One-vs-Rest)")
        plt.legend()

        return plt.gcf()

# =====================================================
# 5️⃣ Residual Plot (Regression)
# =====================================================

def plot_residuals(y_test, preds):

    residuals = y_test - preds

    fig, ax = plt.subplots()
    ax.scatter(preds, residuals)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")

    return fig
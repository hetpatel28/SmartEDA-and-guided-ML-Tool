import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def train_models(df, target, problem_type, selected_models, test_size=0.2):

    df_copy = df.copy()

    # ============================================
    # 1️⃣ Drop rows with missing target
    # ============================================
    initial_rows = len(df_copy)
    df_copy = df_copy.dropna(subset=[target])
    dropped_rows = initial_rows - len(df_copy)

    # ============================================
    # 2️⃣ Encode target if classification
    # ============================================
    if problem_type == "Classification":
        if df_copy[target].dtype in ["object", "category"]:
            le = LabelEncoder()
            df_copy[target] = le.fit_transform(df_copy[target])

    # ============================================
    # 3️⃣ Split X and y
    # ============================================
    X = df_copy.drop(columns=[target])
    y = df_copy[target]

    # ============================================
    # 4️⃣ Auto sample large dataset
    # ============================================
    MAX_ROWS = 120000
    if len(X) > MAX_ROWS:
        sampled = df_copy.sample(MAX_ROWS, random_state=42)
        y = sampled[target]
        X = sampled.drop(columns=[target])

    rows = len(X)

    # ============================================
    # 5️⃣ Safe categorical handling (Memory Safe)
    # ============================================

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    rows = len(X)

    for col in categorical_cols:

        unique_count = X[col].nunique()

        # If too many categories → label encode only
        if unique_count > 20:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # If small categories → one hot but limit features
        elif unique_count <= 20:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)

            # Prevent explosion
            if dummies.shape[1] <= 15:
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
    # ============================================
    # Convert to float32 (Memory Optimization)
    # ============================================

    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")

    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("int32")
    

    # ============================================
    # 6️⃣ Prevent feature explosion
    # ============================================
    MAX_FEATURES = 2000
    if X.shape[1] > MAX_FEATURES:
        X = X.select_dtypes(include=["int64", "float64"])
        X = X.iloc[:, :MAX_FEATURES]

    # ============================================
    # 7️⃣ Handle missing values
    # ============================================
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    X = X.fillna(0)

    # ============================================
    # 8️⃣ Train/Test split
    # ============================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    trained_models = {}
    predictions = {}
    training_times = {}

    # ============================================
    # 9️⃣ Disable heavy models for large datasets
    # ============================================
    if rows > 80000:
        if "Support Vector Machine" in selected_models:
            selected_models.remove("Support Vector Machine")

        if "K-Nearest Neighbors" in selected_models:
            selected_models.remove("K-Nearest Neighbors")

    # ============================================
    # 🔟 Optimized Model Training
    # ============================================
    for model_name in selected_models:

        start_time = time.time()

        if problem_type == "Classification":

            if model_name == "Logistic Regression":
                model = LogisticRegression(
                    max_iter=200,
                    solver="lbfgs",
                    n_jobs=-1
                )

            elif model_name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=30,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=42
                )

            elif model_name == "Gradient Boosting":
                model = HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=8,
                    random_state=42
                )

            elif model_name == "Support Vector Machine":
                model = SVC(kernel="linear")

            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=5)

        else:  # Regression

            if model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=42
                )

            elif model_name == "Gradient Boosting Regressor":
                model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )

            elif model_name == "Ridge Regression":
                model = Ridge()

            elif model_name == "Lasso Regression":
                model = Lasso(max_iter=2000)

        model.fit(X_train, y_train)

        end_time = time.time()
        training_times[model_name] = round(end_time - start_time, 2)

        trained_models[model_name] = model
        predictions[model_name] = model.predict(X_test)

    total_time = round(sum(training_times.values()), 2)

    return (
        trained_models,
        predictions,
        X_test,
        y_test,
        X,
        y,
        dropped_rows,
        training_times,
        total_time
    )
import pandas as pd
import numpy as np


# =====================================================
# 1️⃣ MISSING VALUE FUNCTIONS
# =====================================================

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count.values,
        "Missing %": missing_pct.round(2).values
    })

    return summary.sort_values(by="Missing %", ascending=False)


def get_missing_percentage(df):
    return (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)


def get_missing_matrix(df):
    return df.isnull()


def handle_missing_values(df, column, method, custom_value=None):

    if method == "Drop Rows":
        df = df.dropna(subset=[column])

    elif method == "Drop Column":
        df = df.drop(columns=[column])

    elif method == "Mean":
        df[column] = df[column].fillna(df[column].mean())

    elif method == "Median":
        df[column] = df[column].fillna(df[column].median())

    elif method == "Mode":
        df[column] = df[column].fillna(df[column].mode()[0])

    elif method == "Forward Fill":
        df[column] = df[column].ffill()

    elif method == "Backward Fill":
        df[column] = df[column].bfill()

    elif method == "Custom Value":
        df[column] = df[column].fillna(custom_value)

    return df


# =====================================================
# 2️⃣ DUPLICATE FUNCTIONS
# =====================================================

def get_duplicate_summary(df):
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    return duplicate_count, round(duplicate_pct, 2)


def remove_duplicates(df):
    return df.drop_duplicates()


# =====================================================
# 3️⃣ OUTLIER FUNCTIONS
# =====================================================

def detect_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    return outliers.index.tolist(), lower, upper


def detect_outliers_zscore(df, column, threshold=3):

    mean = df[column].mean()
    std = df[column].std()

    z_scores = (df[column] - mean) / std

    outliers = df[np.abs(z_scores) > threshold]

    return outliers.index.tolist()


def handle_outliers(df, column, outlier_indices, method):

    if method == "Remove":
        df = df.drop(index=outlier_indices)

    elif method == "Cap":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower, upper)

    elif method == "Replace with Median":
        median = df[column].median()
        df.loc[outlier_indices, column] = median

    return df


# =====================================================
# 4️⃣ CONSTANT COLUMN FUNCTIONS
# =====================================================

def detect_constant_columns(df):
    return [col for col in df.columns if df[col].nunique() <= 1]


def remove_columns(df, columns):
    return df.drop(columns=columns)


# =====================================================
# 5️⃣ HIGH CARDINALITY
# =====================================================

def detect_high_cardinality(df, threshold=0.7):
    high_card_cols = []

    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > threshold:
            high_card_cols.append(col)

    return high_card_cols


# =====================================================
# 6️⃣ DATA TYPE MISMATCH
# =====================================================

def detect_dtype_mismatch(df):

    mismatch_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_numeric(df[col])
                mismatch_cols.append((col, "Numeric stored as Object"))
            except:
                try:
                    pd.to_datetime(df[col])
                    mismatch_cols.append((col, "Date stored as Object"))
                except:
                    pass

    return mismatch_cols
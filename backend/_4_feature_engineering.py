import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# =====================================================
# 1️⃣ CREATE NEW FEATURE
# =====================================================

def create_feature(df, col1, col2, operation, new_name):

    if new_name in df.columns:
        return df, False

    if operation == "Add":
        df[new_name] = df[col1] + df[col2]

    elif operation == "Subtract":
        df[new_name] = df[col1] - df[col2]

    elif operation == "Multiply":
        df[new_name] = df[col1] * df[col2]

    elif operation == "Divide":
        df[new_name] = df[col1] / df[col2].replace(0, np.nan)

    return df, True


# =====================================================
# 2️⃣ SKEW TRANSFORMATION
# =====================================================

def transform_skew(df, column, method):

    if method == "Log":
        df[column] = np.log1p(df[column])
    elif method == "Square Root":
        df[column] = np.sqrt(df[column])

    return df, True


# =====================================================
# 3️⃣ ENCODING
# =====================================================

def label_encode(df, columns):

    if not columns:
        return df, False

    le = LabelEncoder()

    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df, True


def one_hot_encode(df, columns):

    if not columns:
        return df, False

    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df, True


# =====================================================
# 4️⃣ BINNING
# =====================================================

def bin_continuous(df, column, bins, method):

    new_col = column + "_binned"

    if new_col in df.columns:
        return df, False

    if method == "Equal Width":
        df[new_col] = pd.cut(df[column], bins=bins)
    elif method == "Equal Frequency":
        df[new_col] = pd.qcut(df[column], q=bins, duplicates='drop')

    return df, True


# =====================================================
# 5️⃣ SCALING
# =====================================================

def scale_features(df, columns, method):

    if not columns:
        return df, False

    if method == "StandardScaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[columns] = scaler.fit_transform(df[columns])

    return df, True


# =====================================================
# 6️⃣ REMOVE HIGH CORRELATION
# =====================================================

def remove_high_correlation(df, threshold=0.9):

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr_matrix = numeric_df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if not to_drop:
        return df, []

    df = df.drop(columns=to_drop)

    return df, to_drop


# =====================================================
# 7️⃣ REMOVE LOW VARIANCE
# =====================================================

def remove_low_variance(df, threshold=0.01):

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    variances = numeric_df.var()

    low_var_cols = variances[variances < threshold].index.tolist()

    if not low_var_cols:
        return df, []

    df = df.drop(columns=low_var_cols)

    return df, low_var_cols
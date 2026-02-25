import pandas as pd
from ydata_profiling import ProfileReport


def get_basic_overview(df: pd.DataFrame) -> dict:
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    missing_ratio = (missing_cells / total_cells) * 100
    duplicate_ratio = (duplicate_rows / df.shape[0]) * 100

    score = 100 - ((missing_ratio * 0.5) + (duplicate_ratio * 0.5))

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "quality_score": round(score, 2),
        "missing_pct": round(missing_ratio, 2),
        "duplicate_pct": round(duplicate_ratio, 2)
    }


def get_data_preview(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    return df.head(rows)


def get_attribute_info(df: pd.DataFrame) -> pd.DataFrame:
    
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    return info_df


def get_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:

    return df.describe().T


def get_categorical_value_counts(df: pd.DataFrame) -> dict:
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    result = {}

    for col in categorical_columns:
        counts = df[col].value_counts(dropna=False)
        percentages = (counts / len(df)) * 100

        value_count_df = pd.DataFrame({
            col: counts.index,
            "Count": counts.values,
            "Percentage": percentages.round(2).values
        })

        result[col] = value_count_df

    return result

def suggest_categorical_columns(df, threshold=20):
    suggested_columns = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)

        if unique_count <= threshold:
            suggested_columns.append({
                "column": col,
                "unique_count": unique_count,
                "current_dtype": str(df[col].dtype)
            })

    return suggested_columns

def get_full_report(df):
    profile = ProfileReport(df, title="YData Profiling Report", explorative=True)
    return profile
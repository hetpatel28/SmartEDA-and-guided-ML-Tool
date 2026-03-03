import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1 UNIVARIATE ANALYSIS
def plot_numeric_univariate(df, column):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig

def plot_categorical_univariate(df, column):
    fig, ax = plt.subplots(figsize=(6,4))
    df[column].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Category Count of {column}")
    return fig


# 2 BIVARIATE ANALYSIS
def plot_numeric_numeric(df, col1, col2):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
    ax.set_title(f"{col1} vs {col2}")
    return fig

def plot_categorical_numeric(df, cat_col, num_col):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
    ax.set_title(f"{num_col} grouped by {cat_col}")
    return fig

def plot_categorical_categorical(df, col1, col2):
    cross_tab = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cross_tab, annot=True, fmt="d", ax=ax)
    ax.set_title(f"{col1} vs {col2}")
    return fig


# 3 CORRELATION ANALYSIS
def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def strong_correlations(df, threshold=0.8):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr_matrix = numeric_df.corr().abs()

    strong_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                strong_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                )

    return strong_pairs


# 4 TARGET BASED ANALYSIS
def target_correlation(df, target):
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()[target].sort_values(ascending=False)
    return corr

def target_group_analysis(df, target, feature):
    return df.groupby(target)[feature].mean()


# 5 PAIRPLOT (LIMITED)
def pairplot_selected(df, columns):
    fig = sns.pairplot(df[columns])
    return fig.fig
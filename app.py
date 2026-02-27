import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import time

# Import backend modules
from backend._1_data_loader import *
from backend._2_data_inspection import *
from backend._3_relation import *
from backend._4_feature_engineering import *
from backend._6_ml_recommendation import *
from backend._7_model_training import *
from backend._8_evaluation import *
from backend._9_explainability import *
from backend._10_report_generator import *



st.set_page_config(page_title="📊 SmartEDA", layout="wide")

st.markdown("<h1 style='text-align:center;color:#42A5EB;'>📊 SmartEDA</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#42A5EB;'>An AI-assisted AutoEDA and guided ML tool</h3>", unsafe_allow_html=True)

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if 'current_section' not in st.session_state:
    st.session_state.current_section = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'df' not in st.session_state:
    st.session_state.df = None
if "df_history" not in st.session_state:
    st.session_state.df_history = []
if "redo_stack" not in st.session_state:
    st.session_state.redo_stack = []
if "change_log" not in st.session_state:
    st.session_state.change_log = []
# Store original dataset
if "original_df" not in st.session_state:
    st.session_state.original_df = None
# Version tracking
if "data_version" not in st.session_state:
    st.session_state.data_version = 1
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "ml_config" not in st.session_state:
    st.session_state.ml_config = None

def log_action(action_text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.change_log.append(
        f"[{timestamp}] (v{st.session_state.data_version}) {action_text}"
    )

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

def get_sample_df(df, max_rows=5000):
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df

#------------------------------------
#data preview and full report function
#------------------------------------
def render_data_preview():
    df = st.session_state.df

    # Basic Overview (Always Visible)
    overview = get_basic_overview(df)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Rows", overview["rows"])
    col2.metric("Columns", overview["columns"])
    col3.metric("Data Quality Score", f'{overview["quality_score"]}%')
    col4.metric("Missing %", f'{overview["missing_pct"]}%')
    col5.metric("Duplicate %", f'{overview["duplicate_pct"]}%')

    st.divider()

    # Dropdown Section
    analysis_option = st.selectbox(
        "Select Analysis Section",
        [
            "Data Preview",
            "Attribute Information",
            "Descriptive Statistics",
            "Categorical Value Counts",
            "Smart Categorical Suggestions",
            "Full Profiling Report"
        ]
    )

    # Data Preview
    if analysis_option == "Data Preview":
        st.dataframe(get_data_preview(df))

    # Attribute Info
    elif analysis_option == "Attribute Information":
        st.dataframe(get_attribute_info(df))

    # Descriptive Statistics
    elif analysis_option == "Descriptive Statistics":
        st.dataframe(get_descriptive_statistics(df))

    # Categorical Value Counts
    elif analysis_option == "Categorical Value Counts":
        counts = get_categorical_value_counts(df)

        for column_name, table in counts.items():
            st.subheader(column_name)
            st.dataframe(table)

    # Smart Categorisation
    elif analysis_option == "Smart Categorical Suggestions":

        threshold = st.number_input(
            "Set Unique Value Threshold",
            min_value=2,
            max_value=100,
            value=20
        )

        suggestions = suggest_categorical_columns(df, threshold)

        if suggestions:
            for item in suggestions:
                col_name = item["column"]

                convert = st.checkbox(
                    f"Convert '{col_name}' (Unique: {item['unique_count']}) to category"
                )

                if convert:
                    df[col_name] = df[col_name].astype("category")

            st.session_state.df = df
            st.success("Selected columns converted successfully.")
        else:
            st.info("No columns meet the threshold.")

    # Full Report
    elif analysis_option == "Full Profiling Report":
        st.subheader("📋 Full Profiling Report")
        if len(df) > 100000:
            st.warning("Large dataset detected. Profiling may take few time.")
        profile = get_full_report(df)
        st_profile_report(profile)

#----------------------------------------
#data cleaning and preprocessing function
#----------------------------------------
def render_data_cleaning():

    df = st.session_state.df

    # ==============================
    # Data Version + Quality Score
    # ==============================
    st.info(f"Current Data Version: v{st.session_state.data_version}")

    overview = get_basic_overview(df)
    st.metric("Data Quality Score", f'{overview["quality_score"]}%')

    # ==============================
    # Undo / Redo / Reset Controls
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Undo Last Action"):
            if st.session_state.df_history:
                st.session_state.redo_stack.append(df.copy())
                st.session_state.df = st.session_state.df_history.pop()

                if st.session_state.data_version > 1:
                    st.session_state.data_version -= 1

                log_action("Undo performed")
                st.success("Undo successful.")
            else:
                st.warning("No previous state available.")

    with col2:
        if st.button("Redo Last Action"):
            if st.session_state.redo_stack:
                st.session_state.df_history.append(df.copy())
                st.session_state.df = st.session_state.redo_stack.pop()

                st.session_state.data_version += 1

                log_action("Redo performed")
                st.success("Redo successful.")
            else:
                st.warning("No redo state available.")

    with col3:
        if st.button("Reset to Original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.df_history = []
            st.session_state.redo_stack = []
            st.session_state.data_version = 1
            log_action("Dataset reset to original upload")
            st.success("Dataset reset successfully.")

    # ==============================
    # Cleaning Options
    # ==============================
    cleaning_option = st.selectbox(
        "Select Inspection Action",
        [
            "Missing Value Analysis",
            "Duplicate Analysis",
            "Outlier Detection",
            "Constant Column Detection",
            "High Cardinality Check",
            "Data Type Review"
        ]
    )

    # =====================================================
    # 1️⃣ Missing Values
    # =====================================================
    if cleaning_option == "Missing Value Analysis":

        st.dataframe(get_missing_summary(df))

        if st.checkbox("Show Missing Heatmap"):
            plt.figure(figsize=(10,4))
            sns.heatmap(df.isnull(), cbar=False)
            st.pyplot(plt.gcf())
            plt.clf()

        if st.checkbox("Show Missing % Chart"):
            missing_pct = get_missing_percentage(df)
            plt.figure(figsize=(10,4))
            missing_pct.plot(kind="bar")
            plt.ylabel("Missing %")
            st.pyplot(plt.gcf())
            plt.clf()

        column = st.selectbox("Select Column", df.columns)

        method = st.selectbox(
            "Select Method",
            ["Drop Rows", "Drop Column", "Mean", "Median",
             "Mode", "Forward Fill", "Backward Fill", "Custom Value"]
        )

        custom_value = None
        if method == "Custom Value":
            custom_value = st.text_input("Enter Custom Value")

        if st.button("Apply Missing Handling"):

            st.session_state.df_history.append(df.copy())
            st.session_state.redo_stack.clear()

            df = handle_missing_values(df, column, method, custom_value)
            st.session_state.df = df

            st.session_state.data_version += 1
            log_action(f"Missing handled on '{column}' using '{method}'")

            st.success("Missing values handled successfully.")

    # =====================================================
    # 2️⃣ Duplicate Analysis
    # =====================================================
    elif cleaning_option == "Duplicate Analysis":

        duplicate_count, duplicate_pct = get_duplicate_summary(df)
        st.write(f"Duplicate Rows: {duplicate_count} ({duplicate_pct}%)")

        if st.button("Remove Duplicates"):

            st.session_state.df_history.append(df.copy())
            st.session_state.redo_stack.clear()

            df = remove_duplicates(df)
            st.session_state.df = df

            st.session_state.data_version += 1
            log_action("Removed duplicate rows")

            st.success("Duplicates removed successfully.")

    # =====================================================
    # 3️⃣ Outlier Detection
    # =====================================================
    elif cleaning_option == "Outlier Detection":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        column = st.selectbox("Select Numeric Column", numeric_cols)
        method = st.selectbox("Detection Method", ["IQR", "Z-Score"])

        if method == "IQR":
            outlier_indices, lower, upper = detect_outliers_iqr(df, column)
        else:
            outlier_indices = detect_outliers_zscore(df, column)

        st.write(f"Outliers Detected: {len(outlier_indices)}")

        if st.checkbox("Show Boxplot"):
            plt.figure(figsize=(6,4))
            sns.boxplot(x=df[column])
            st.pyplot(plt.gcf())
            plt.clf()

        if st.checkbox("Show Scatter Plot"):
            plt.figure(figsize=(6,4))
            plt.scatter(range(len(df[column])), df[column])
            st.pyplot(plt.gcf())
            plt.clf()

        action = st.selectbox(
            "Select Handling Method",
            ["Remove", "Cap", "Replace with Median"]
        )

        if st.button("Apply Outlier Handling"):

            st.session_state.df_history.append(df.copy())
            st.session_state.redo_stack.clear()

            df = handle_outliers(df, column, outlier_indices, action)
            st.session_state.df = df

            st.session_state.data_version += 1
            log_action(f"Outliers handled on '{column}' using '{action}'")

            st.success("Outliers handled successfully.")

    # =====================================================
    # 4️⃣ Constant Column Detection
    # =====================================================
    elif cleaning_option == "Constant Column Detection":

        constant_cols = detect_constant_columns(df)

        if constant_cols:
            selected = st.multiselect("Select Columns to Remove", constant_cols)

            if st.button("Remove Selected Columns"):

                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                df = remove_columns(df, selected)
                st.session_state.df = df

                st.session_state.data_version += 1
                log_action("Removed constant columns")

                st.success("Columns removed successfully.")
        else:
            st.success("No constant columns found.")

    # =====================================================
    # 5️⃣ High Cardinality
    # =====================================================
    elif cleaning_option == "High Cardinality Check":

        high_card_cols = detect_high_cardinality(df)

        if high_card_cols:
            st.warning("High Cardinality Columns Detected:")
            st.write(high_card_cols)
        else:
            st.success("No high cardinality columns found.")

    # =====================================================
    # 6️⃣ Data Type Review
    # =====================================================
    elif cleaning_option == "Data Type Review":

        mismatches = detect_dtype_mismatch(df)

        if mismatches:
            for col, issue in mismatches:
                st.warning(f"{col}: {issue}")
        else:
            st.success("No data type mismatch found.")

    # ==============================
    # Change Log Panel
    # ==============================
    st.divider()
    st.subheader("Change Log")

    log_text = "No changes have been made."

    if st.session_state.change_log:
        for entry in reversed(st.session_state.change_log):
            st.write(entry)

        log_text = "\n".join(st.session_state.change_log)
    else:
        st.info("No actions performed yet.")

    st.download_button(
        label="Download Change Log",
        data=log_text,
        file_name="change_log.txt",
        mime="text/plain"
    )

# ------------------------------------------
# pattern and relationship discovery function
# ------------------------------------------
def render_pattern_discovery():

    df = st.session_state.df

    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Univariate Analysis",
            "Bivariate Analysis",
            "Correlation Heatmap",
            "Target-Based Analysis",
            "Strong Relationship Detector",
            "Pairplot (Limited Columns)"
        ]
    )

    # =====================================================
    # 1️⃣ UNIVARIATE
    # =====================================================
    if analysis_type == "Univariate Analysis":

        column = st.selectbox("Select Column", df.columns)

        if df[column].dtype in ["int64", "float64"]:
            fig = plot_numeric_univariate(df, column)
        else:
            fig = plot_categorical_univariate(df, column)

        st.pyplot(fig)

        st.download_button(
            label="Download Plot",
            data=fig_to_bytes(fig),
            file_name="univariate_plot.png",
            mime="image/png"
        )

        # =====================================================
        # 2️⃣ BIVARIATE
        # =====================================================
    elif analysis_type == "Bivariate Analysis":

        col1 = st.selectbox("Select First Column", df.columns)
        col2 = st.selectbox("Select Second Column", df.columns)

        dtype1 = df[col1].dtype
        dtype2 = df[col2].dtype

        df_sample = get_sample_df(df)

        # =====================================================
        # CASE 1: Numeric - Numeric
        # =====================================================
        if dtype1 in ["int64", "float64"] and dtype2 in ["int64", "float64"]:

            with st.spinner("Generating scatter plot..."):
                fig = plot_numeric_numeric(df_sample, col1, col2)

            st.pyplot(fig)
            st.download_button(
                label="Download Plot",
                data=fig_to_bytes(fig),
                file_name="numeric_numeric_plot.png",
                mime="image/png"
            )

            # =====================================================
            # CASE 2: Categorical - Numeric
            # =====================================================
        elif (dtype1 == "object" and dtype2 in ["int64", "float64"]) or \
            (dtype2 == "object" and dtype1 in ["int64", "float64"]):

            # Identify which is categorical
            cat_col = col1 if dtype1 == "object" else col2
            num_col = col2 if dtype1 == "object" else col1

            unique_categories = df[cat_col].nunique()

            if unique_categories > 30:
                st.warning(
                    f"'{cat_col}' has {unique_categories} categories. "
                    "Too many for safe boxplot visualization."
                )
            else:
                with st.spinner("Generating boxplot..."):
                    fig = plot_categorical_numeric(df_sample, cat_col, num_col)

                st.pyplot(fig)
                st.download_button(
                    label="Download Plot",
                    data=fig_to_bytes(fig),
                    file_name="categorical_numeric_plot.png",
                    mime="image/png"
                )

            # =====================================================
            # CASE 3: Categorical - Categorical
            # =====================================================
        else:

            unique1 = df[col1].nunique()
            unique2 = df[col2].nunique()

            matrix_size = unique1 * unique2

            if unique1 > 30 or unique2 > 30:
                st.warning(
                    f"Too many categories ({unique1} x {unique2}). "
                    "Heatmap disabled to prevent freezing."
                )

            elif matrix_size > 2000:
                st.warning(
                    f"Crosstab size ({matrix_size} cells) too large. "
                    "Heatmap disabled for safety."
                )

            else:
                with st.spinner("Generating categorical heatmap..."):
                    fig = plot_categorical_categorical(df_sample, col1, col2)

                st.pyplot(fig)
                st.download_button(
                    label="Download Plot",
                    data=fig_to_bytes(fig),
                    file_name="categorical_categorical_plot.png",
                    mime="image/png"
                )

    # =====================================================
    # 3️⃣ CORRELATION HEATMAP
    # =====================================================
    elif analysis_type == "Correlation Heatmap":

        fig = correlation_heatmap(df)
        st.pyplot(fig)

        st.download_button(
            label="Download Heatmap",
            data=fig_to_bytes(fig),
            file_name="correlation_heatmap.png",
            mime="image/png"
        )

    # =====================================================
    # 4️⃣ TARGET BASED ANALYSIS
    # =====================================================
    elif analysis_type == "Target-Based Analysis":

        target = st.selectbox("Select Target Column", df.columns)

        if df[target].dtype in ["int64", "float64"]:
            corr = target_correlation(df, target)
            st.dataframe(corr)

        feature = st.selectbox("Select Feature for Group Analysis", df.columns)
        grouped = target_group_analysis(df, target, feature)
        st.dataframe(grouped)

    # =====================================================
    # 5️⃣ STRONG RELATIONSHIP
    # =====================================================
    elif analysis_type == "Strong Relationship Detector":

        threshold = st.slider("Correlation Threshold", 0.5, 1.0, 0.8)
        pairs = strong_correlations(df, threshold)

        if pairs:
            for col1, col2, val in pairs:
                st.write(f"{col1} - {col2}: {round(val,2)}")
        else:
            st.info("No strong correlations found.")

    # =====================================================
    # 6️⃣ PAIRPLOT
    # =====================================================
    elif analysis_type == "Pairplot (Limited Columns)":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        selected = st.multiselect("Select up to 3 numeric columns", numeric_cols, max_selections=3)

        if selected:
            fig = pairplot_selected(df, selected)
            st.pyplot(fig)

            st.download_button(
                label="Download Pairplot",
                data=fig_to_bytes(fig),
                file_name="pairplot.png",
                mime="image/png"
            )

# ----------------------------
# feature engineering function
# ----------------------------
def render_feature_engineering():

    # if not isinstance(st.session_state.df, pd.DataFrame):
    #     st.error("Dataset corrupted. Please reset to original.")
    #     return

    df = st.session_state.df

    st.info(f"Current Data Version: v{st.session_state.data_version}")

    # ==============================
    # Undo / Redo / Reset Controls
    # ==============================
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Undo Last Action"):
            if st.session_state.df_history:
                st.session_state.redo_stack.append(df.copy())
                st.session_state.df = st.session_state.df_history.pop()

                if st.session_state.data_version > 1:
                    st.session_state.data_version -= 1

                log_action("Undo performed")
                st.success("Undo successful.")
            else:
                st.warning("No previous state available.")

    with col2:
        if st.button("Redo Last Action"):
            if st.session_state.redo_stack:
                st.session_state.df_history.append(df.copy())
                st.session_state.df = st.session_state.redo_stack.pop()

                st.session_state.data_version += 1
                log_action("Redo performed")
                st.success("Redo successful.")
            else:
                st.warning("No redo state available.")

    with col3:
        if st.button("Reset to Original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.df_history = []
            st.session_state.redo_stack = []
            st.session_state.data_version = 1
            log_action("Dataset reset to original upload")
            st.success("Dataset reset successfully.")

    task = st.selectbox(
        "Select Engineering Task",
        [
            "Create New Feature",
            "Transform Skewed Data",
            "Encode Categorical Variables",
            "Bin Continuous Variable",
            "Scale Features",
            "Remove Highly Correlated Features",
            "Remove Low Variance Features"
        ]
    )

    # =====================================================
    # 1️⃣ CREATE FEATURE
    # =====================================================
    if task == "Create New Feature":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        col1 = st.selectbox("Select First Column", numeric_cols)
        col2 = st.selectbox("Select Second Column", numeric_cols)
        operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])
        new_name = st.text_input("New Feature Name")

        if st.button("Create Feature"):

            df_new, changed = create_feature(df.copy(), col1, col2, operation, new_name)

            if changed:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Created feature '{new_name}'")

                st.success("Feature created successfully.")
            else:
                st.warning("Feature not created (name exists or invalid).")

    # =====================================================
    # 2️⃣ SKEW TRANSFORMATION
    # =====================================================
    elif task == "Transform Skewed Data":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        column = st.selectbox("Select Column", numeric_cols)
        method = st.selectbox("Method", ["Log", "Square Root"])

        if st.button("Apply Transformation"):

            df_new, changed = transform_skew(df.copy(), column, method)

            if changed:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Applied {method} transform on '{column}'")

                st.success("Transformation applied.")
            else:
                st.info("No transformation applied.")

    # =====================================================
    # 3️⃣ ENCODING
    # =====================================================
    elif task == "Encode Categorical Variables":

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        selected = st.multiselect("Select Columns", cat_cols)
        method = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding"])

        if st.button("Apply Encoding"):

            if method == "Label Encoding":
                df_new, changed = label_encode(df.copy(), selected)
            else:
                df_new, changed = one_hot_encode(df.copy(), selected)

            if changed:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Applied {method}")

                st.success("Encoding applied.")
            else:
                st.info("No encoding applied.")

    # =====================================================
    # 4️⃣ BINNING
    # =====================================================
    elif task == "Bin Continuous Variable":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        column = st.selectbox("Select Column", numeric_cols)
        bins = st.number_input("Number of Bins", min_value=2, max_value=20, value=4)
        method = st.selectbox("Method", ["Equal Width", "Equal Frequency"])

        if st.button("Apply Binning"):

            df_new, changed = bin_continuous(df.copy(), column, bins, method)

            if changed:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Binned '{column}' into {bins} bins")

                st.success("Binning applied.")
            else:
                st.info("Binning not applied (column may already exist).")

    # =====================================================
    # 5️⃣ SCALING
    # =====================================================
    elif task == "Scale Features":

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        selected = st.multiselect("Select Columns", numeric_cols)
        method = st.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler"])

        if st.button("Apply Scaling"):

            df_new, changed = scale_features(df.copy(), selected, method)

            if changed:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Applied {method}")

                st.success("Scaling applied.")
            else:
                st.info("No scaling applied.")

    # =====================================================
    # 6️⃣ REMOVE HIGH CORRELATION
    # =====================================================
    elif task == "Remove Highly Correlated Features":

        threshold = st.slider("Correlation Threshold", 0.5, 1.0, 0.9)

        if st.button("Remove Correlated Features"):

            df_new, dropped = remove_high_correlation(df.copy(), threshold)

            if dropped:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Removed correlated features: {dropped}")

                st.success(f"Removed: {dropped}")
            else:
                st.info("No correlated features above threshold.")

    # =====================================================
    # 7️⃣ VARIANCE THRESHOLD
    # =====================================================
    elif task == "Remove Low Variance Features":

        threshold = st.number_input("Variance Threshold", value=0.01)

        if st.button("Remove Low Variance"):

            df_new, dropped = remove_low_variance(df.copy(), threshold)

            if dropped:
                st.session_state.df_history.append(df.copy())
                st.session_state.redo_stack.clear()

                st.session_state.df = df_new
                st.session_state.data_version += 1
                log_action(f"Removed low variance features: {dropped}")

                st.success(f"Removed: {dropped}")
            else:
                st.info("No low variance features found.")

    st.divider()
    st.subheader("Change Log")

    log_text = "No changes have been made."

    if st.session_state.change_log:
        for entry in reversed(st.session_state.change_log):
            st.write(entry)

        log_text = "\n".join(st.session_state.change_log)
    else:
        st.info("No actions performed yet.")

    st.download_button(
        label="Download Change Log",
        data=log_text,
        file_name="change_log.txt",
        mime="text/plain"
    )

# --------------------------------------------
# save processed dataset and download function
# --------------------------------------------
def render_save_processed_dataset():

    df = st.session_state.df

    st.info(f"Current Data Version: v{st.session_state.data_version}")

    # ==============================
    # Dataset Summary
    # ==============================
    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("Preview of Processed Dataset:")
    st.dataframe(df.head())

    st.divider()
    # change log
    st.subheader("Change Log")

    log_text = "No changes have been made."

    if st.session_state.change_log:
        for entry in reversed(st.session_state.change_log):
            st.write(entry)

        log_text = "\n".join(st.session_state.change_log)
    else:
        st.info("No actions performed yet.")

    st.download_button(
        label="Download Change Log",
        data=log_text,
        file_name="change_log.txt",
        mime="text/plain"
    )
    st.divider()

    # ==============================
    # File Name Input
    # ==============================
    file_name = st.text_input("Enter File Name (without extension)", "processed_dataset")

    # ==============================
    # CSV Download
    # ==============================
    csv_data = df.to_csv(index=False).encode("utf-8")

    if st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name=f"{file_name}.csv",
        mime="text/csv"
    ):
        log_action("Processed dataset downloaded as CSV")

    st.success("Dataset ready for Machine Learning phase.")

# ----------------------------------------
# machine learning recommendation function
# ----------------------------------------
def render_ml_recommendation():

    df = st.session_state.df

    st.info(f"Current Data Version: v{st.session_state.data_version}")

    st.subheader("Step 1: Select Target Column")

    target = st.selectbox("Select Target Variable", df.columns)

    if target:

        # Detect problem type
        problem_type = detect_problem_type(df, target)

        st.success(f"Detected Problem Type: {problem_type}")

        # Suggest models
        suggested_models = suggest_models(problem_type)

        st.subheader("Step 2: Suggested Models")

        selected_models = st.multiselect(
            "Select Model(s) to Train",
            suggested_models
        )

        # Suggest metrics
        suggested_metrics = suggest_metrics(problem_type)

        st.subheader("Step 3: Suggested Evaluation Metrics")
        st.write(", ".join(suggested_metrics))

        st.divider()

        # Store configuration
        if st.button("Confirm ML Configuration"):

            st.session_state.ml_config = {
                "target": target,
                "problem_type": problem_type,
                "models": selected_models,
                "metrics": suggested_metrics
            }

            log_action(
                f"ML Configuration Set: Target='{target}', "
                f"Type='{problem_type}', Models={selected_models}"
            )

            st.success("Configuration saved. Ready for training phase.")

# ----------------------------------------
# model training and prediction function
# ----------------------------------------


def render_model_training():

    if "ml_config" not in st.session_state:
        st.warning("Please complete ML Recommendation phase first.")
        return

    config = st.session_state.ml_config
    df = st.session_state.df

    st.info(f"Problem Type: {config['problem_type']}")
    st.write(f"Target Variable: {config['target']}")
    st.write(f"Selected Models: {config['models']}")

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

    if st.button("Train Selected Models"):

        if not config["models"]:
            st.warning("Please select at least one model.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_total = time.time()

        trained_models = {}
        predictions = {}
        training_times = {}

        models = config["models"].copy()
        total_models = len(models)

        for i, model_name in enumerate(models):

            status_text.text(f"Training {model_name}...")

            single_config = {
                "problem_type": config["problem_type"],
                "target": config["target"],
                "models": [model_name]
            }

            result = train_models(
                df,
                config["target"],
                config["problem_type"],
                [model_name],
                test_size
            )

            (
                tm,
                preds,
                X_test,
                y_test,
                X_processed,
                y_processed,
                dropped_rows,
                model_times,
                _
            ) = result

            trained_models.update(tm)
            predictions.update(preds)
            training_times.update(model_times)

            progress_bar.progress((i + 1) / total_models)

        end_total = time.time()
        total_time = round(end_total - start_total, 2)

        st.session_state.trained_models = trained_models
        st.session_state.predictions = predictions
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_processed = X_processed
        st.session_state.y_processed = y_processed

        status_text.text("Training completed.")

        st.success("Models trained successfully!")

        st.subheader("Training Time Summary")

        for model, t in training_times.items():
            st.write(f"{model}: {t} seconds")

        st.write(f"Total Training Time: {total_time} seconds")

        st.subheader("Prediction Preview")

        for model_name, preds in predictions.items():
            st.write(f"Model: {model_name}")
            st.write(preds[:10])

# -------------------------
# model evaluation function
# -------------------------
def render_model_evaluation():

    if "trained_models" not in st.session_state:
        st.warning("Please complete Model Training phase first.")
        return

    config = st.session_state.ml_config
    trained_models = st.session_state.trained_models
    predictions = st.session_state.predictions
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    st.info(f"Problem Type: {config['problem_type']}")

    # =====================================================
    # 1️⃣ Model Comparison Table
    # =====================================================
    st.subheader("Model Comparison")

    results_df = evaluate_models(
        config["problem_type"],
        trained_models,
        predictions,
        y_test
    )
    st.session_state.evaluation_results = results_df

    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Evaluation Results",
        data=csv,
        file_name="model_evaluation.csv",
        mime="text/csv"
    )

    # =====================================================
    # 2️⃣ Detailed Analysis
    # =====================================================
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(trained_models.keys())
    )

    model = trained_models[selected_model]
    preds = predictions[selected_model]

    if config["problem_type"] == "Classification":

        if st.checkbox("Show Confusion Matrix"):
            fig = plot_confusion_matrix(y_test, preds)
            st.pyplot(fig)

        if st.checkbox("Show ROC Curve"):
            if not hasattr(model, "predict_proba"):
                st.warning("ROC curve not supported for this model.")
            else:
                fig = plot_roc_curve(model, X_test, y_test)
                st.pyplot(fig)
    else:

        if st.checkbox("Show Residual Plot"):
            fig = plot_residuals(y_test, preds)
            st.pyplot(fig)

    # =====================================================
    # 3️⃣ Cross Validation
    # =====================================================
    if st.checkbox("Perform Cross Validation"):

        X_full = st.session_state.X_processed
        y_full = st.session_state.y_processed

        cv_score = perform_cross_validation(
            config["problem_type"],
            model,
            X_full,
            y_full
        )

        st.success(f"Cross Validation Score: {cv_score}")

# ------------------------------------
# explainability and insights function
# ------------------------------------
def render_explainability():

    if "trained_models" not in st.session_state:
        st.warning("Please complete Model Training first.")
        return

    if st.session_state.X_processed is None:
        st.warning("Please complete Model Training first.")
        return

    trained_models = st.session_state.trained_models
    X_processed = st.session_state.X_processed
    config = st.session_state.ml_config

    st.subheader("Model Explainability & Insights")

    selected_model_name = st.selectbox(
        "Select Model",
        list(trained_models.keys())
    )

    model = trained_models[selected_model_name]

    # =====================================================
    # Feature Importance
    # =====================================================
    importance_df = get_feature_importance(model, X_processed.columns)

    if importance_df is not None:

        st.subheader("Feature Importance")
        st.dataframe(importance_df.head(20))

        fig, ax = plt.subplots()
        ax.barh(
            importance_df.head(10)["Feature"],
            importance_df.head(10)["Importance"]
        )
        ax.invert_yaxis()
        st.pyplot(fig)

        insight_text = generate_insights(importance_df, config["problem_type"])
        st.info(insight_text)

    # =====================================================
    # Linear Coefficients
    # =====================================================
    coef_df = get_coefficients(model, X_processed.columns)

    if coef_df is not None:

        st.subheader("Coefficient Importance")
        st.dataframe(coef_df.head(20))

        fig, ax = plt.subplots()
        ax.barh(
            coef_df.head(10)["Feature"],
            coef_df.head(10)["Coefficient"]
        )
        ax.invert_yaxis()
        st.pyplot(fig)

        insight_text = generate_insights(coef_df, config["problem_type"])
        st.info(insight_text)

    # =====================================================
    # SHAP (Optional)
    # =====================================================
    if hasattr(model, "feature_importances_"):

        if st.checkbox("Generate SHAP Summary (May take time)"):

            # Sample for performance
            X_sample = X_processed.sample(
                min(2000, len(X_processed)),
                random_state=42
            )

            with st.spinner("Generating SHAP explanation..."):
                fig = generate_shap_summary(model, X_sample)

            st.pyplot(fig)

# --------------------------------
# final report generation function
# --------------------------------
def render_final_report():

    if st.session_state.evaluation_results is None:
        st.warning("Please complete Model Evaluation first.")
        return

    df = st.session_state.df
    evaluation_df = st.session_state.evaluation_results
    config = st.session_state.ml_config
    trained_models = st.session_state.trained_models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    dataset_name = st.session_state.uploaded_file.name

    if st.button("Generate Report"):

        file_path = "SmartEDA_Report.pdf"

        overview = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "quality_score": get_basic_overview(df)["quality_score"]
        }

        generate_final_report(
            file_path,
            dataset_name,
            df,
            overview,
            evaluation_df,
            config["problem_type"],
            trained_models,
            X_test,
            y_test
        )

        with open(file_path, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="SmartEDA_Report.pdf",
                mime="application/pdf"
            )

        st.success("report generated successfully.")

#-----------------------------------
# Function to render section content
#-----------------------------------
def render_section_content(section_name):
    """Render content based on selected section using dictionary routing"""

    SECTION_FUNCTIONS = {
        "📊 Data Preview And Full report":render_data_preview ,
        "🧹 Data cleaning and Preprocessing":render_data_cleaning ,
        "🔍 Pattern & Relationship Discovery":render_pattern_discovery ,
        "⚙️ Feature Engineering":render_feature_engineering ,
        "💾 Save Processed Dataset & Download":render_save_processed_dataset,
        "🤖 Guided Machine Learning Recommendation":render_ml_recommendation,
        "🎯 Model Training & Prediction":render_model_training,
        "📈 Model Evaluation":render_model_evaluation,
        "💡 Explainability & Insights":render_explainability,
        "📋 Report Generation":render_final_report
    }

    selected_function = SECTION_FUNCTIONS.get(section_name)

    if selected_function:
        selected_function()


# Sidebar controls
st.sidebar.header("🔧 Controls")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Only show section selection after a file is uploaded
if uploaded_file is not None:
    if st.session_state.uploaded_file != uploaded_file:
        try:
            # Load dataframe using backend loader
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            
            st.session_state.df_history = []
            st.session_state.redo_stack = []
            st.session_state.change_log = []
            st.session_state.data_version = 1

            st.session_state.uploaded_file = uploaded_file

        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    sections = [
        "📊 Data Preview And Full report",
        "🧹 Data cleaning and Preprocessing",
        "🔍 Pattern & Relationship Discovery",
        "⚙️ Feature Engineering",
        "💾 Save Processed Dataset & Download",
        "🤖 Guided Machine Learning Recommendation",
        "🎯 Model Training & Prediction",
        "📈 Model Evaluation",
        "💡 Explainability & Insights",
        "📋 Report Generation"
    ]
    
    selected_section = st.sidebar.radio("📊 Choose Section", sections, key="section_radio")
    st.session_state.current_section = selected_section
    
    st.sidebar.divider()

    st.divider()
    render_section_content(selected_section)
    
else:
    # Welcome screen when no file is uploaded
    st.sidebar.info("⬆️ Upload a CSV file to get started")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("🚀 Getting Started")
        st.write("""
        ### Steps to use SmartEDA:
        1. **Upload** your CSV file using the uploader
        2. **Select** a section from the sidebar
        3. **Analyze** your data step by step
        4. **Export** results and reports
        """)
    
    with col2:
        st.header("✨ Features")
        st.write("""
        - 📊 Automated EDA
        - 🧹 Data Cleaning
        - 🤖 ML Recommendations
        - 📈 Model Training
        - 💡 Explainability
        - 📋 Report Generation
        """)
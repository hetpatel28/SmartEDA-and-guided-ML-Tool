from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix


# STYLING
def get_custom_styles():
    styles = getSampleStyleSheet()

    custom_heading = ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#1F4E79"),
        spaceAfter=10
    )

    return styles, custom_heading


# PERFORMANCE GRAPH
def generate_performance_graph(evaluation_df, problem_type, file_path):
    plt.figure(figsize=(6,4))

    if problem_type == "Regression":
        metric = "R2 Score"
    else:
        metric = "Accuracy"

    plt.bar(evaluation_df["Model"], evaluation_df[metric])
    plt.xticks(rotation=30)
    plt.ylabel(metric)
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


# DATASET DISTRIBUTION CHART
def generate_dataset_chart(df, file_path):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns[:3]

    if len(numeric_cols) == 0:
        return None

    plt.figure(figsize=(6,4))

    for col in numeric_cols:
        sns.histplot(df[col], kde=True)

    plt.title("Sample Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return file_path


# CONFUSION MATRIX
def generate_confusion_matrix_plot(model, X_test, y_test, file_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


# SHAP PLOT
def generate_shap_plot(model, X_test, file_path):
    try:
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        return True
    except:
        return False


# AUTO INSIGHTS
def generate_model_insights(evaluation_df, problem_type):
    insights = []

    for _, row in evaluation_df.iterrows():
        model_name = row["Model"]

        if problem_type == "Regression":
            score = row["R2 Score"]
            if score < 0:
                text = f"{model_name} shows weak predictive performance."
            elif score < 0.5:
                text = f"{model_name} demonstrates moderate performance."
            else:
                text = f"{model_name} demonstrates strong predictive capability."
        else:
            score = row["Accuracy"]
            if score < 0.6:
                text = f"{model_name} accuracy is relatively low."
            elif score < 0.85:
                text = f"{model_name} performs reasonably well."
            else:
                text = f"{model_name} achieves strong classification accuracy."

        insights.append(text)

    return insights


# FINAL MULTI-PAGE REPORT
def generate_final_report(
    file_path,
    dataset_name,
    df,
    overview,
    evaluation_df,
    problem_type,
    trained_models,
    X_test,
    y_test
):

    doc = SimpleDocTemplate(file_path)
    elements = []
    styles, custom_heading = get_custom_styles()

    timestamp = datetime.now().strftime("%d %B %Y, %H:%M")

    # Determine best model
    if problem_type == "Regression":
        metric = "R2 Score"
    else:
        metric = "Accuracy"

    best_row = evaluation_df.sort_values(by=metric, ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_score = best_row[metric]
    best_model_obj = trained_models[best_model_name]

    # PAGE 1 — EXECUTIVE SUMMARY
    elements.append(Paragraph("SmartEDA Professional ML Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("1. Executive Summary", custom_heading))

    summary_text = (
        f"Dataset: {dataset_name}<br/>"
        f"Records: {overview['rows']}<br/>"
        f"Features: {overview['columns']}<br/>"
        f"Problem Type: {problem_type}<br/>"
        f"Best Model: {best_model_name}<br/>"
        f"Best {metric}: {round(best_score,4)}"
    )
    elements.append(Paragraph(summary_text, styles["Normal"]))
    elements.append(PageBreak())


    # PAGE 2 — DATASET ANALYSIS
    elements.append(Paragraph("2. Dataset Analysis", custom_heading))

    dataset_chart_path = "dataset_chart.png"
    if generate_dataset_chart(df, dataset_chart_path):
        elements.append(Image(dataset_chart_path, width=5*inch, height=3*inch))

    elements.append(PageBreak())


    # PAGE 3 — MODEL EVALUATION
    elements.append(Paragraph("3. Model Evaluation", custom_heading))

    eval_data = [evaluation_df.columns.tolist()] + evaluation_df.values.tolist()

    table = Table(eval_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#DCE6F1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    performance_path = "performance.png"
    generate_performance_graph(evaluation_df, problem_type, performance_path)
    elements.append(Image(performance_path, width=5*inch, height=3*inch))

    elements.append(PageBreak())


    # PAGE 4 — MODEL INSIGHTS
    elements.append(Paragraph("4. Model Insights", custom_heading))

    insights = generate_model_insights(evaluation_df, problem_type)
    for text in insights:
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    elements.append(PageBreak())


    # PAGE 5 — CONFUSION MATRIX (Classification Only)
    if problem_type != "Regression":
        elements.append(Paragraph("5. Confusion Matrix", custom_heading))

        cm_path = "confusion_matrix.png"
        generate_confusion_matrix_plot(best_model_obj, X_test, y_test, cm_path)
        elements.append(Image(cm_path, width=4*inch, height=4*inch))

        elements.append(PageBreak())


    # PAGE 6 — SHAP EXPLAINABILITY
    elements.append(Paragraph("6. SHAP Explainability", custom_heading))

    shap_path = "shap_plot.png"
    if generate_shap_plot(best_model_obj, X_test, shap_path):
        elements.append(Image(shap_path, width=5*inch, height=3*inch))
    else:
        elements.append(Paragraph("SHAP visualization not supported for this model.", styles["Normal"]))

    elements.append(Spacer(1, 0.5 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Report Generated On: {timestamp}", styles["Normal"]))
    elements.append(Paragraph("Generated using SmartEDA AI Platform", styles["Normal"]))

    doc.build(elements)

    # Cleanup
    for f in ["dataset_chart.png", "performance.png", "confusion_matrix.png", "shap_plot.png"]:
        if os.path.exists(f):
            os.remove(f)
# utils/pdf_report.py
from fpdf import FPDF
from PIL import Image
import json
import textwrap
import os
import pandas as pd

def _add_wrapped_text(pdf, text, max_width=90, line_height=5):
    # simple wrapper for long paragraphs
    wrapper = textwrap.TextWrapper(width=120)
    lines = wrapper.wrap(text=text)
    for line in lines:
        pdf.multi_cell(0, line_height, line)
    pdf.ln(1)

def generate_pdf_report(
    out_path: str,
    project_title: str,
    rfm_path: str,
    rfm_clv_path: str,
    regression_results_json: str,
    saved_images: dict,
    segment_labels_json: str = None
):
    """
    Generate a PDF report containing project summary, model explanations,
    results table, and visualization images.

    Parameters
    ----------
    out_path: path to output PDF file
    project_title: Title string
    rfm_path: path to rfm_with_segments.csv
    rfm_clv_path: path to rfm_clv.csv
    regression_results_json: models/regression_results.json
    saved_images: dict of {label: path_to_png} in the order you want them in PDF
    segment_labels_json: optional path to cluster_labels.json
    """
    # Load data
    try:
        results = pd.read_json(regression_results_json)
    except Exception:
        results = None

    try:
        rfm = pd.read_csv(rfm_path)
    except Exception:
        rfm = None

    try:
        rfm_clv = pd.read_csv(rfm_clv_path)
    except Exception:
        rfm_clv = None

    try:
        labels = json.load(open(segment_labels_json)) if segment_labels_json and os.path.exists(segment_labels_json) else None
    except Exception:
        labels = None

    pdf = FPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=12)

    # --- Cover page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 10, project_title, ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 6, "Customer Analytics Pipeline — Segmentation (RFM + KMeans) and CLV (90 days) Regression Models")
    pdf.ln(4)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 7, "Project Summary", ln=True)
    pdf.set_font("Arial", size=11)
    summary = (
        "This report summarizes the Customer Analytics project which performs RFM feature engineering "
        "from transaction data, segments customers using KMeans clustering, and predicts 90-day Customer "
        "Lifetime Value (CLV) using multiple regression models. The report includes model comparisons, "
        "visualizations and segment-level insights for data-driven marketing actions."
    )
    _add_wrapped_text(pdf, summary)

    # --- Model explanations ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 7, "Models Used (Regression) — Short Explanation", ln=True)
    pdf.set_font("Arial", size=11)
    model_texts = {
        "Linear Regression": "A baseline linear model that assumes linear relation between features and target.",
        "Ridge Regression": "Linear regression with L2 regularization to reduce overfitting and handle multicollinearity.",
        "Lasso Regression": "Linear regression with L1 regularization, encourages sparse coefficients (feature selection).",
        "Random Forest": "Ensemble of decision trees using bagging; captures non-linear relationships and interactions.",
        "Extra Trees": "Similar to Random Forest but uses randomized splits; often reduces variance and training time.",
        "Gradient Boosting": "Boosted trees (sequential) that combine weak learners into a strong model (good for tabular data).",
        "XGBoost": "Optimized gradient boosting implementation with regularization and advanced features; strong performance on tabular data."
    }
    for name, desc in model_texts.items():
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, f"- {name}", ln=True)
        pdf.set_font("Arial", size=10)
        _add_wrapped_text(pdf, desc, line_height=5)

    # --- Model results table ---
    if results is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 7, "Model Comparison (Validation)", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=10)
        # Table header
        pdf.cell(60, 6, "Model", border=1)
        pdf.cell(40, 6, "MAE", border=1)
        pdf.cell(40, 6, "RMSE", border=1)
        pdf.cell(40, 6, "R2", border=1)
        pdf.ln()
        for _, row in results.iterrows():
            pdf.cell(60, 6, str(row.get('model','')), border=1)
            pdf.cell(40, 6, f"{row.get('mae',0):.2f}", border=1)
            pdf.cell(40, 6, f"{row.get('rmse',0):.2f}", border=1)
            pdf.cell(40, 6, f"{row.get('r2',0):.3f}", border=1)
            pdf.ln()
    else:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 7, "Model Comparison: Not available", ln=True)

    # --- Visualizations ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 7, "Key Visualizations", ln=True)
    pdf.ln(4)

    for label, img_path in saved_images.items():
        if not os.path.exists(img_path):
            continue
        # add title
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 6, label, ln=True)
        pdf.ln(1)

        # open image and resize to fit width
        try:
            img = Image.open(img_path)
            w, h = img.size
            # A4 printable width ~ 190mm minus margins; convert px to mm roughly via DPI assumption 96
            max_width_mm = 180
            dpi = img.info.get('dpi', (96,96))[0]
            width_mm = w / dpi * 25.4
            height_mm = h / dpi * 25.4
            scale = min(1.0, max_width_mm / max(1e-6, width_mm))
            w_mm = width_mm * scale
            h_mm = height_mm * scale
            pdf.image(img_path, w=w_mm)
            pdf.ln(6)
        except Exception:
            # fallback, place image raw
            pdf.image(img_path, w=160)
            pdf.ln(6)

    # --- Segment summary (top 5 segments by size) ---
    if rfm is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 7, "Segment Summary", ln=True)
        pdf.ln(4)
        seg_counts = rfm['segment_name'].value_counts().reset_index()
        seg_counts.columns = ['segment_name', 'count']
        pdf.set_font("Arial", size=10)
        for _, r in seg_counts.head(10).iterrows():
            pdf.cell(0, 6, f"{r['segment_name']}: {int(r['count']):,} customers", ln=True)

    # --- Footer ---
    pdf.add_page()
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 6, "Report generated by Customer Analytics Pipeline", ln=True)
    pdf.cell(0, 6, "Includes segmentation (RFM + KMeans) and CLV model comparison (multiple regressors)", ln=True)

    pdf.output(out_path)
    return out_path

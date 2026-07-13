# app/streamlit_app_modern.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from datetime import datetime
from src.pdf_report import generate_pdf_report

# -------- CONFIG ----------
DATA_DIR = "data/processed"
MODEL_DIR = "models"
IMG_DIR = "artifacts/imgs"
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)

# -------- LOAD DATA & MODELS ----------
rfm = pd.read_csv(os.path.join(DATA_DIR, "rfm_with_segments.csv"))
rfm_clv = pd.read_csv(os.path.join(DATA_DIR, "rfm_clv.csv"))
results_df = pd.read_json(os.path.join(MODEL_DIR, "regression_results.json"))

# pick clv_model file
clv_model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("clv_model")]
clv_model = joblib.load(os.path.join(MODEL_DIR, clv_model_files[0])) if clv_model_files else None

kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
cluster_labels = None
labels_path = os.path.join(MODEL_DIR, "cluster_labels.json")
if os.path.exists(labels_path):
    cluster_labels = pd.read_json(labels_path, typ='series').to_dict()

# -------- PAGE CONFIG & STYLES ----------
st.set_page_config(page_title="Customer Analytics — Modern Dashboard", layout="wide")

# inject modern CSS
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f6f8fb 0%, #ffffff 100%); }
    .big-header { font-size:28px; font-weight:700; color:#0f172a; }
    .kpi { background: linear-gradient(90deg,#ffffff,#f7fbff); border-radius:12px; padding:14px; box-shadow: 0 2px 8px rgba(15,23,42,0.06); }
    .card-title { font-size:12px; color:#6b7280; }
    .card-value { font-size:20px; font-weight:700; color:#0f172a }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
with st.sidebar:
    # st.image("path/to/your/logo.png", width=200)  # placeholder for logo
    st.title("Customer Analytics")
    st.write("RFM Segmentation • CLV Modeling")
    page = st.radio("Navigation", ["Segmentation", "CLV", "Model Comparison", "Explorer"])
    st.markdown("---")
    st.write("Export")
    report_btn = st.button("📄 Generate PDF Report")

# ---------- UTIL: KPI CARD ----------
def kpi_card(title, value, delta=None):
    st.markdown(f"""
    <div class="kpi">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- PAGE: Segmentation ----------
if page == "Segmentation":
    st.markdown('<div class="big-header">Customer Segmentation</div>', unsafe_allow_html=True)
    # KPI row
    total_customers = len(rfm)
    num_segments = rfm['segment'].nunique()
    champions = int((rfm['segment_name'] == 'Champions').sum()) if 'segment_name' in rfm.columns else 0

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        kpi_card("Total Customers", f"{total_customers:,}")
    with c2:
        kpi_card("Active Segments", f"{num_segments}")
    with c3:
        kpi_card("Champions", f"{champions:,}")

    # Distribution
    st.markdown("#### Segment Distribution")
    seg_counts = rfm['segment_name'].value_counts().reset_index()
    seg_counts.columns = ['segment','count']
    fig = px.pie(seg_counts, names='segment', values='count', hole=0.4, 
                 color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig, use_container_width=True)
    # save image
    fig.write_image(os.path.join(IMG_DIR, 'segment_distribution.png'), scale=2)

    # RFM scatter
    st.markdown("#### Recency vs Monetary")
    sc = px.scatter(rfm, x='recency', y='monetary_total', color='segment_name', hover_data=['frequency'],
                    color_discrete_sequence=px.colors.qualitative.Plotly, title="Recency vs Monetary by Segment")
    st.plotly_chart(sc, use_container_width=True)
    sc.write_image(os.path.join(IMG_DIR, 'recency_monetary_scatter.png'), scale=2)

    # segment profile table
    st.markdown("#### Segment Profiles (Averages)")
    st.dataframe(rfm.groupby('segment_name')[['recency','frequency','monetary_total']].mean().round(2))
    # save small CSV snapshot for report
    rfm.groupby('segment_name')[['recency','frequency','monetary_total']].mean().round(2).to_csv(os.path.join(IMG_DIR, 'segment_profiles.csv'))

# ---------- PAGE: CLV ----------
elif page == "CLV":
    st.markdown('<div class="big-header">Customer Lifetime Value (CLV)</div>', unsafe_allow_html=True)

    X = rfm_clv[['recency','frequency','monetary_total','monetary_avg','monetary_variance']].fillna(0)
    y_true = rfm_clv['clv_90'].fillna(0).values
    y_pred = clv_model.predict(X) if clv_model is not None else np.zeros(len(X))
    rfm_clv['predicted_clv'] = y_pred

    total_pred_rev = int(rfm_clv['predicted_clv'].sum())
    avg_clv = round(rfm_clv['predicted_clv'].mean(),2)
    high_value = int((rfm_clv['predicted_clv'] > avg_clv*2).sum())

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        kpi_card("Predicted 90-day Revenue", f"${total_pred_rev:,}")
    with c2:
        kpi_card("Average Predicted CLV", f"${avg_clv}")
    with c3:
        kpi_card("High-value Cust.", f"{high_value:,}")

    # CLV histogram
    st.markdown("#### Predicted CLV Distribution by Segment")
    box_fig = px.box(rfm_clv, x='segment_name', y='predicted_clv', 
                     color='segment_name', points="all",
                     title='Predicted CLV Distribution by Segment',
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(box_fig, use_container_width=True)
    box_fig.write_image(os.path.join(IMG_DIR, 'clv_distribution.png'), scale=2)

    # Actual vs predicted
    st.markdown("#### Actual vs Predicted CLV")
    ap = px.scatter(x=y_true, y=y_pred, labels={'x':'Actual CLV','y':'Predicted CLV'}, title='Actual vs Predicted CLV')
    st.plotly_chart(ap, use_container_width=True)
    ap.write_image(os.path.join(IMG_DIR, 'clv_actual_vs_pred.png'), scale=2)

    st.markdown("#### Top 20 Predicted CLV Customers")
    st.dataframe(rfm_clv.sort_values('predicted_clv', ascending=False).head(20))

# ---------- PAGE: Model Comparison ----------
elif page == "Model Comparison":
    st.markdown('<div class="big-header">Model Comparison</div>', unsafe_allow_html=True)

    # show table
    st.dataframe(results_df)

    # MAE chart
    fig_mae = px.line(results_df.sort_values('mae'), x='model', y='mae', title='MAE by Model', markers=True, color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_mae, use_container_width=True)
    fig_mae.write_image(os.path.join(IMG_DIR, 'model_mae.png'), scale=2)

    # R2 chart
    fig_r2 = px.line(results_df.sort_values('r2', ascending=False), x='model', y='r2', title='R² by Model', markers=True, color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_r2, use_container_width=True)
    fig_r2.write_image(os.path.join(IMG_DIR, 'model_r2.png'), scale=2)

# ---------- PAGE: Explorer ----------
else:
    st.markdown('<div class="big-header">Customer Explorer</div>', unsafe_allow_html=True)
    seg_opts = ['All'] + sorted(rfm['segment_name'].unique().tolist())
    sel = st.selectbox("Filter by segment", seg_opts)
    view = rfm_clv.copy()
    if sel != 'All':
        view = view[view['segment_name'] == sel]
    st.dataframe(view)
    csv_data = view.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_data, file_name=f"customers_{sel}.csv")

# ---------- Generate PDF if requested ----------
if report_btn:
    st.info("Generating PDF report... this may take a few seconds.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pdf = os.path.join("artifacts", f"customer_report_{timestamp}.pdf")
    os.makedirs("artifacts", exist_ok=True)

    saved_images = {
        "Segment Distribution": os.path.join(IMG_DIR, 'segment_distribution.png'),
        "Recency vs Monetary": os.path.join(IMG_DIR, 'recency_monetary_scatter.png'),
        "CLV Distribution": os.path.join(IMG_DIR, 'clv_distribution.png'),
        "CLV Actual vs Predicted": os.path.join(IMG_DIR, 'clv_actual_vs_pred.png'),
        "Model MAE Comparison": os.path.join(IMG_DIR, 'model_mae.png'),
        "Model R2 Comparison": os.path.join(IMG_DIR, 'model_r2.png'),
    }

    # Fallback: ensure images exist (if not, generate minimal ones)
    for k,v in saved_images.items():
        if not os.path.exists(v):
            # create an empty placeholder image if missing
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB',(1200,700),(255,255,255))
            d = ImageDraw.Draw(img)
            d.text((50,50), f"Missing: {os.path.basename(v)}", fill=(100,100,100))
            img.save(v)

    # call PDF generator
    pdf_path = generate_pdf_report(
        out_path=out_pdf,
        project_title="Customer Analytics — Segmentation & CLV Report",
        rfm_path=os.path.join(DATA_DIR, "rfm_with_segments.csv"),
        rfm_clv_path=os.path.join(DATA_DIR, "rfm_clv.csv"),
        regression_results_json=os.path.join(MODEL_DIR, "regression_results.json"),
        saved_images=saved_images,
        segment_labels_json=os.path.join(MODEL_DIR, "cluster_labels.json")
    )

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Report", f, file_name=os.path.basename(pdf_path))
    st.success("PDF generated and ready to download.")

# 🛒 Customer Analytics Pipeline — Segmentation & CLV Prediction

[![Project Status](https://img.shields.io/badge/Project-Customer_Analytics-blue?style=flat-square)](https://github.com/kaustubhthorat07/customer-analytics-pipeline)
[![Tech Stack](https://img.shields.io/badge/Tech-Python_|_Sklearn_|_XGBoost_|_Streamlit-success?style=flat-square)](https://github.com/kaustubhthorat07/customer-analytics-pipeline)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)](https://github.com/kaustubhthorat07/customer-analytics-pipeline)

---

## 📘 Overview

The **Customer Analytics Pipeline** is a complete, end-to-end machine learning workflow that converts raw customer transaction data into:
- **Customer Segmentation**: Actionable behavioral groupings using K-Means.
- **Behavioral Insights**: Multi-dimensional RFM profile analysis.
- **90-Day CLV Predictions**: Predictive revenue modeling using regressor models.
- **Interactive Dashboards**: Real-time analytical dashboard built with Streamlit and Plotly.
- **Automated PDF Reports**: Multi-page business-ready reporting.

It follows a modular, production-ready architecture designed for enterprise analytics and robust business deployments.

---

## 🖼️ Dashboard Preview

The interactive dashboard provides key performance indicators (KPIs), segmentation views, CLV forecasting, and a customer explorer utility.

| Page 1: Customer Segmentation | Page 2: CLV Prediction & Actuals |
|:---:|:---:|
| ![Segment Distribution](image1.png) | ![CLV Distribution](image2.png) |

| Page 3: Recency vs Monetary | Page 4: Model Performance |
|:---:|:---:|
| ![Recency vs Monetary](image3.png) | ![Model Comparison](image4.png) |

---

## 🚀 Key Features

### 1. RFM Feature Engineering
Aggregates transactional histories to automatically extract:
*   **Recency**: Days since last purchase.
*   **Frequency**: Total number of transactions.
*   **Monetary Metrics**: Total spend, average purchase value, and monetary variance.

### 2. Smart Customer Segmentation (K-Means)
Identifies distinct behavioral customer segments using standardized scaling:
*   **Champions**: Highly active, frequent, and high-value customers.
*   **Loyal**: Reliable customers with consistent engagement.
*   **Potential Loyalists**: Recent shoppers with high frequency growth potential.
*   **At Risk**: Customers showing warning signs of churn.
*   **Lost**: Dormant accounts requiring re-engagement campaigns.

### 3. Predictive CLV Modeling
Trains, validates, and compares regression algorithms including:
*   Linear Regression, Ridge, and Lasso (baselines)
*   Ensemble models: Random Forest, Extra Trees, and Gradient Boosting
*   Extreme Gradient Boosting (XGBoost)

*Note: The pipeline automatically evaluates validation Mean Absolute Error (MAE) and selects the best-performing model for serialization.*

### 4. Interactive Streamlit Dashboard
*   **KPI Cards**: Immediate summary of predicted revenue, average CLV, and high-value customer counts.
*   **Customer Explorer**: Searchable and filterable data grid with CSV exporter.
*   **One-Click PDF Export**: Generate publication-quality reports instantly.

### 5. Automated PDF Reporting
Generates a structured, multi-page business document containing:
*   Executive Summary
*   Detailed Segment Profiles (Averages)
*   Validation Metric Comparison Tables
*   Embedded Visualization Charts

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kaustubhthorat07/customer-analytics-pipeline.git
cd customer-analytics-pipeline
```

### 2. Create and Activate Virtual Environment
*   **Windows (PowerShell)**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
*   **macOS / Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Place your raw transaction CSV/Excel file in:
```text
data/raw/online_retail_II.csv
```

---

## 🏃‍♂️ Usage Guide

### 1. Run the End-to-End Pipeline
Executes feature extraction, K-Means clustering, CLV label calculation, and regression comparison:
```bash
python main_pipeline.py
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 3. Generate Report
Inside the Streamlit sidebar, click **📄 Generate PDF Report** to download the auto-compiled PDF report immediately.

---

## ✉️ Contact

For questions, feedback, or collaboration opportunities, feel free to contact:

*   **Developer**: Kaustubh Thorat
*   **Email**: [kaustubhthorat07@gmail.com](mailto:kaustubhthorat07@gmail.com)

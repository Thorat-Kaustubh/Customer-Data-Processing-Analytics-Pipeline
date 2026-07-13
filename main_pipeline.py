import os
import pandas as pd
import numpy as np

from src.rfm_engineering import create_rfm
from src.clustering import scale_features, find_elbow_inertia, find_best_silhouette, train_kmeans
from src.clv_regression import create_clv_label, train_regressors
from src.utils import save_model, save_json

# ----------------------------------------
# CONFIG
# ----------------------------------------
DATA_FILE = "data/raw/online_retail_II.csv"   # <-- UPDATE THIS TO YOUR FILE
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------------
# 1. LOAD DATA
# ----------------------------------------
print("\n📥 Loading dataset...")
df = pd.read_csv(DATA_FILE, encoding="latin1")
print("Loaded shape:", df.shape)

# Normalize column names
df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
print("\n🔎 Columns detected:", df.columns.tolist())

# ----------------------------------------
# 2. AUTO-DETECT DATE COLUMN
# ----------------------------------------
possible_date_cols = [
    "invoicedate", "invoice_date", "invoice_datetime",
    "date", "order_date", "purchase_date", "transaction_date"
]

date_col = None
for col in possible_date_cols:
    if col in df.columns:
        date_col = col
        break

if date_col is None:
    raise ValueError(f"❌ No date column found. Available columns: {df.columns.tolist()}")

print(f"\n📅 Using date column: {date_col}")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df.rename(columns={date_col: "date"}, inplace=True)

# ----------------------------------------
# 3. AUTO-DETECT CUSTOMER ID
# ----------------------------------------
possible_customer_cols = ["customer_id", "customerid", "cust_id", "user_id", "client_id"]

customer_col = None
for col in possible_customer_cols:
    if col in df.columns:
        customer_col = col
        break

if customer_col is None:
    print("⚠ No customer ID column found → generating synthetic IDs")
    df["customer_id"] = df.groupby(df.columns[0]).ngroup()
else:
    df.rename(columns={customer_col: "customer_id"}, inplace=True)

# ----------------------------------------
# 4. AUTO-DETECT AMOUNT/PRICE COLUMN
# ----------------------------------------
possible_amount_cols = ["amount", "unit_price", "price", "value", "sales", "total"]

amount_col = None
for col in possible_amount_cols:
    if col in df.columns:
        amount_col = col
        break

# If price*quantity exists → use it
if "unit_price" in df.columns and "quantity" in df.columns:
    print("🧮 Creating amount = unit_price × quantity")
    df["amount"] = df["unit_price"] * df["quantity"]

elif amount_col:
    print(f"💰 Using existing amount column: {amount_col}")
    df.rename(columns={amount_col: "amount"}, inplace=True)

else:
    raise ValueError("❌ Could not detect amount/price column in dataset.")

# ----------------------------------------
# 5. CREATE RFM TABLE
# ----------------------------------------
print("\n📊 Creating RFM table...")
rfm, snapshot_date = create_rfm(df)
rfm.to_csv(f"{PROCESSED_DIR}/rfm_table.csv", index=False)
print("✔ RFM table saved.")

# ----------------------------------------
# 6. RUN KMEANS CLUSTERING
# ----------------------------------------
print("\n🤖 Training KMeans clustering...")

feature_cols = ["recency", "frequency", "monetary_total"]
X = rfm[feature_cols].fillna(0)

# Scale features
X_scaled, scaler = scale_features(X)
save_model(scaler, f"{MODELS_DIR}/scaler.pkl")

# Elbow method (optional)
k_range, inertias = find_elbow_inertia(X_scaled)
print("Inertia values:", inertias)

# Silhouette scores (optional)
sil_scores = find_best_silhouette(X_scaled)
print("Silhouette scores:", sil_scores)

# Choose number of clusters
n_clusters = 5
km = train_kmeans(X_scaled, n_clusters=n_clusters)

# Assign clusters
rfm["segment"] = km.labels_
save_model(km, f"{MODELS_DIR}/kmeans_model.pkl")

# Segment name mapping
segment_labels = {
    0: "Champions",
    1: "Loyal",
    2: "Potential Loyalists",
    3: "At Risk",
    4: "Lost"
}

rfm["segment_name"] = rfm["segment"].map(segment_labels)
save_json(segment_labels, f"{MODELS_DIR}/cluster_labels.json")

rfm.to_csv(f"{PROCESSED_DIR}/rfm_with_segments.csv", index=False)
print("✔ KMeans clustering completed.")

# ----------------------------------------
# 7. CREATE CLV (90-DAY FUTURE VALUE)
# ----------------------------------------
print("\n💰 Creating CLV (90 days) target...")
clv = create_clv_label(df, snapshot_date=snapshot_date)

rfm_clv = rfm.merge(clv, on="customer_id", how="left").fillna(0)
rfm_clv.to_csv(f"{PROCESSED_DIR}/rfm_clv.csv", index=False)

print("✔ CLV target created.")

# ----------------------------------------
# 8. TRAIN CLV REGRESSION MODELS
# ----------------------------------------
print("\n📈 Training CLV regression models...")

X_reg = rfm_clv[["recency", "frequency", "monetary_total", "monetary_avg", "monetary_variance"]]
y_reg = rfm_clv["clv_90"]

results, best_key, best_model = train_regressors(X_reg, y_reg)

print("\n📌 Model Results:")
for k, v in results.items():
    print(f"{k}: MAE={v['mae']:.2f}, RMSE={v['rmse']:.2f}, R2={v['r2']:.4f}")

print(f"\n🏆 Best Model: {best_key}")

# Save best model
save_model(best_model, f"{MODELS_DIR}/clv_model_{best_key}.pkl")

# Save regression results to JSON for the app
results_for_json = []
for model_name, metrics in results.items():
    results_for_json.append({
        "model": model_name,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"]
    })
save_json(results_for_json, f"{MODELS_DIR}/regression_results.json")

print("\n🎉 Pipeline completed successfully!")

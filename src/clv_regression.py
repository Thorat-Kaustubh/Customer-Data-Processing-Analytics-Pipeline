# src/clv_regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def create_clv_label(transactions, customer_col='customer_id', date_col='date', amount_col='amount', snapshot_date=None, window_days=90):
    """
    For each customer, compute sum(amount) in the window (snapshot_date, snapshot_date + window_days]
    Return DataFrame with columns: customer_id, clv_90
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df[date_col].max()
    start = snapshot_date + pd.Timedelta(days=1)
    end = snapshot_date + pd.Timedelta(days=window_days)
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    future = df[mask].groupby(customer_col)[amount_col].sum().reset_index().rename(columns={amount_col: f'clv_{window_days}'})
    return future

def train_regressors(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = {}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['rf'] = {
        'model': rf,
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'r2': r2_score(y_test, y_pred_rf)
    }

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=random_state, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['xgb'] = {
        'model': xgb,
        'mae': mean_absolute_error(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'r2': r2_score(y_test, y_pred_xgb)
    }

    # choose best by MAE (lower is better)
    best_key = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_key]['model']
    return results, best_key, best_model

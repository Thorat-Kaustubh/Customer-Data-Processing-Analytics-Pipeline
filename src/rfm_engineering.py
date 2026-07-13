# src/rfm_engineering.py
import pandas as pd

def create_rfm(df, customer_col='customer_id', date_col='date', amount_col='amount', snapshot_date=None):
    """
    Given transactions df, returns rfm dataframe (recency, frequency, monetary_total, monetary_avg, monetary_std).
    snapshot_date should be a pd.Timestamp; if None it uses df[date_col].max() + 1 day
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Use only historical data for RFM
    historical_df = df[df[date_col] < snapshot_date].copy()

    # Aggregate
    agg = historical_df.groupby(customer_col).agg(
        last_purchase=(date_col, 'max'),
        frequency=(date_col, 'count'),
        monetary_total=(amount_col, 'sum'),
        monetary_avg=(amount_col, 'mean'),
        monetary_std=(amount_col, 'std')
    ).reset_index()

    agg['recency'] = (snapshot_date - agg['last_purchase']).dt.days
    agg['monetary_std'] = agg['monetary_std'].fillna(0.0)
    agg['monetary_variance'] = agg['monetary_std'] ** 2

    # reorder/keep columns
    rfm = agg[[customer_col, 'recency', 'frequency', 'monetary_total', 'monetary_avg', 'monetary_std', 'monetary_variance']]
    return rfm, snapshot_date

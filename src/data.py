"""Dataset ingestion and forward return target generation."""

import os
import pandas as pd
import numpy as np


def load_msci_dataset(filepath=None):
    """Load weekly international MSCI indices dataset."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'weekly_msci_indices.csv')
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df


def create_target_and_features(df, target_col='LUXXX', horizon=1, return_type='pct'):
    """Compute weekly returns and align contemporaneous features with 1-period forward target."""
    if return_type == 'pct':
        ret_df = df.pct_change().dropna()
    else:
        ret_df = np.log(df / df.shift(1)).dropna()

    X = ret_df.drop(columns=[target_col], errors='ignore')
    y_reg = ret_df[target_col].shift(-horizon).dropna()
    X = X.iloc[:len(y_reg)]
    return X, y_reg

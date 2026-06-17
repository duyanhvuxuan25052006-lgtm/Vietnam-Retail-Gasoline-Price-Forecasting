import pandas as pd
import numpy as np

MARKET_COLS = ['brent', 'wti', 'dxy', 'vix', 'usdvnd']
TARGETS = ['RON95', 'E5_RON92']

# Lunar New Year (Tết) dates from 2019 to 2026
TET_DATES = [
    pd.Timestamp('2019-02-05'),
    pd.Timestamp('2020-01-25'),
    pd.Timestamp('2021-02-12'),
    pd.Timestamp('2022-02-01'),
    pd.Timestamp('2023-01-22'),
    pd.Timestamp('2024-02-10'),
    pd.Timestamp('2025-01-29'),
    pd.Timestamp('2026-02-17'),
    pd.Timestamp('2027-01-29') 
]

def add_event_dummies(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. COVID-19 dummy (First case Jan 23, 2020 to end of 2021 lockdowns)
    df['dummy_covid'] = ((df['date'] >= '2020-01-23') & (df['date'] <= '2021-12-31')).astype(int)
    
    # 2. Russia-Ukraine war shock dummy (Feb 24, 2022 to end of 2022 oil spikes)
    df['dummy_war'] = ((df['date'] >= '2022-02-24') & (df['date'] <= '2022-12-31')).astype(int)
    
    # 3. Tet Holiday distance and dummy
    days_to_tet = []
    for dt in df['date']:
        # Find absolute difference in days to all Tet dates, take minimum
        diffs = [abs((dt - tet).days) for tet in TET_DATES]
        days_to_tet.append(min(diffs))
        
    df['days_to_tet'] = days_to_tet
    # Tet dummy: 1 if within 14 days of Tet (pre-holiday shopping & holiday)
    df['dummy_tet'] = (df['days_to_tet'] <= 14).astype(int)
    
    return df

def build_lag_and_rolling_features(df):
    """
    Build backward-looking features only.

    Anti-leakage design:
    - Target lags use shift(1)/shift(2) so the model never sees the current price.
    - Market columns in merged_dataset.csv are already aligned to the previous
      trading day (see data_fetcher.merge_and_save), so at retail date t the
      raw market feature is information available before the announcement.
    - Rolling stats are computed on that already-lagged market series; each row
      only aggregates observations from the same row and earlier rows.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    for target in TARGETS:
        df[f'{target}_lag1'] = df[target].shift(1)
        df[f'{target}_lag2'] = df[target].shift(2)

    for col in MARKET_COLS:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        df[f'{col}_ma3'] = df[col].rolling(3, min_periods=3).mean()
        df[f'{col}_ma5'] = df[col].rolling(5, min_periods=5).mean()
        df[f'{col}_std3'] = df[col].rolling(3, min_periods=3).std()

    return df

def get_feature_columns(target):
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}")

    feature_cols = [f'{target}_lag1', f'{target}_lag2']

    for col in MARKET_COLS:
        feature_cols.extend([
            col,
            f'{col}_lag1',
            f'{col}_lag2',
            f'{col}_ma3',
            f'{col}_ma5',
            f'{col}_std3',
        ])

    feature_cols.extend(['dummy_covid', 'dummy_war', 'dummy_tet', 'days_to_tet'])

    if target == 'RON95':
        feature_cols.extend(['tax_bvmt_ron95', 'tax_ttdb_rate_ron95'])
    else:
        feature_cols.extend(['tax_bvmt_e5', 'tax_ttdb_rate_e5'])

    feature_cols.extend(['tax_vat_rate', 'standard_cost', 'standard_profit'])
    return feature_cols

def assert_no_missing_features(df, feature_cols):
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    null_counts = df[feature_cols].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Feature columns contain NaNs: {null_counts.to_dict()}")

def get_winsorize_columns(feature_cols):
    """Columns that should be winsorized inside each CV fold."""
    return [
        c for c in feature_cols
        if 'dummy' not in c and 'tax' not in c and 'standard' not in c
    ]


def apply_winsorizing(train_df, test_df, columns, lower_pct=0.025, upper_pct=0.975):
    """
    Fits winsorization bounds on train_df and applies (clips) them on both train_df and test_df
    to prevent data leakage. Only applied to feature columns, NEVER target Y.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for col in columns:
        if col not in train_df.columns:
            continue
            
        lower_bound = train_df[col].quantile(lower_pct)
        upper_bound = train_df[col].quantile(upper_pct)
        
        train_df[col] = train_df[col].clip(lower_bound, upper_bound)
        if not test_df.empty:
            test_df[col] = test_df[col].clip(lower_bound, upper_bound)
            
    return train_df, test_df

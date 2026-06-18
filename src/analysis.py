import os
import sys
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

from data_fetcher import MARKET_COLS, apply_policy_features
from features import (
    add_event_dummies,
    apply_winsorizing,
    assert_no_missing_features,
    build_lag_and_rolling_features,
    get_feature_columns,
    get_winsorize_columns,
)
from models import calculate_mape, get_splits


def load_raw_dataset():
    path = os.path.join(DATA_DIR, "merged_dataset.csv")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if df['date'].duplicated().any():
        duplicates = df.loc[df['date'].duplicated(), 'date'].dt.strftime('%Y-%m-%d').tolist()
        raise ValueError(f"Duplicate dates found in merged_dataset.csv: {duplicates}")

    return df


def prepare_feature_frame(raw_df):
    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = add_event_dummies(df)
    df = build_lag_and_rolling_features(df)

    required_cols = sorted(set(
        get_feature_columns('RON95')
        + get_feature_columns('E5_RON92')
        + ['RON95', 'E5_RON92']
    ))
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    assert_no_missing_features(df, get_feature_columns('RON95'))
    assert_no_missing_features(df, get_feature_columns('E5_RON92'))
    return df


def weighted_average(values, weights):
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def run_policy_cycle_comparison(raw_df):
    print("\nRunning policy-cycle resampling comparison (7-day vs 10-day)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resample_periods = {'7D': '7-day cycle', '10D': '10-day cycle'}
    cycle_results = {}

    for freq, label in resample_periods.items():
        df_res = (
            raw_df.set_index('date')
                  .resample(freq)
                  .last()
                  .ffill()
                  .reset_index()
        )
        df_res = apply_policy_features(df_res)
        df_res = prepare_feature_frame(df_res)

        feature_cols = get_feature_columns('RON95')
        splits = get_splits(df_res, min_train_obs=40, min_test_obs=8)

        fold_mapes = []
        fold_weights = []
        for split in splits:
            train_df = df_res.iloc[split['train_idx']].copy()
            test_df = df_res.iloc[split['test_idx']].copy()
            train_df, test_df = apply_winsorizing(
                train_df,
                test_df,
                get_winsorize_columns(feature_cols),
            )

            X_train, y_train = train_df[feature_cols], train_df['RON95']
            X_test, y_test = test_df[feature_cols], test_df['RON95']

            model = lgb.LGBMRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_mapes.append(calculate_mape(y_test.values, pred))
            fold_weights.append(split['test_rows'])

        avg_mape = weighted_average(fold_mapes, fold_weights) if fold_mapes else np.nan
        cycle_results[label] = avg_mape
        print(f"Weighted MAPE for {label}: {avg_mape:.2f}%")

    result_df = pd.DataFrame({
        'Resample_Cycle': list(cycle_results.keys()),
        'Weighted_MAPE_RON95': list(cycle_results.values()),
    })
    result_df.to_csv(os.path.join(OUTPUT_DIR, "policy_cycle_comparison.csv"), index=False)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=result_df, x='Resample_Cycle', y='Weighted_MAPE_RON95', color='#2f6f73')
    plt.title('Forecasting Error by Resampled Cycle')
    plt.ylabel('Weighted MAPE (%)')
    plt.xlabel('Resampled Cycle')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontweight='bold')

    plot_path = os.path.join(OUTPUT_DIR, "policy_analysis.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Policy analysis plot saved to {plot_path}")
    return result_df


def run_sensitivity_analysis(feature_df):
    print("\nRunning scenario sensitivity analysis...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feature_cols = get_feature_columns('RON95')
    assert_no_missing_features(feature_df, feature_cols)

    X = feature_df[feature_cols]
    y = feature_df['RON95']

    model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    latest_obs = feature_df.iloc[-1].copy()
    current_brent = latest_obs['brent']
    brent_changes = np.arange(-10, 11, 2)
    bog_interventions = [-1000, -500, 0, 500, 1000]

    grid = np.zeros((len(bog_interventions), len(brent_changes)))
    for i, bog in enumerate(bog_interventions):
        for j, brent_change in enumerate(brent_changes):
            sim_obs = latest_obs.copy()
            shifted_brent = current_brent + brent_change
            sim_obs['brent'] = shifted_brent
            sim_obs['brent_lag1'] = shifted_brent
            sim_obs['brent_ma3'] = shifted_brent
            sim_obs['brent_ma5'] = shifted_brent

            pred_base = model.predict(pd.DataFrame([sim_obs[feature_cols]]))[0]
            grid[i, j] = pred_base - bog

    grid_df = pd.DataFrame(
        grid,
        index=[f"{x:+} VND/L" if x != 0 else "0 VND/L" for x in bog_interventions],
        columns=[f"{x:+} USD" if x != 0 else "0 USD" for x in brent_changes],
    )
    grid_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_matrix.csv"))

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        grid_df,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        cbar_kws={'label': 'Retail RON95 Price (VND/L)'},
    )
    plt.title(f'RON95 Price Sensitivity (Base Brent: {current_brent:.2f} USD/bbl)')
    plt.xlabel('Change in Global Brent Crude Oil Price (USD/barrel)')
    plt.ylabel('BOG Fund Intervention Level')

    plot_path = os.path.join(OUTPUT_DIR, "sensitivity_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Sensitivity heatmap saved to {plot_path}")
    return grid_df


def next_thursday_after(latest_date):
    days_ahead = (3 - latest_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return latest_date + pd.Timedelta(days=days_ahead)


def load_market_snapshot(next_date):
    yfinance_path = os.path.join(DATA_DIR, "yfinance_features.csv")
    market_df = pd.read_csv(yfinance_path)
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df.sort_values('date').reset_index(drop=True)

    cutoff = next_date - pd.Timedelta(days=1)
    available = market_df[market_df['date'] <= cutoff].dropna(subset=MARKET_COLS)
    if available.empty:
        raise ValueError(f"No market data available on or before {cutoff.date()}.")

    return available.iloc[-1].copy()


def build_forecast_row(raw_df, next_date):
    market_snapshot = load_market_snapshot(next_date)

    future_row = raw_df.iloc[-1].copy()
    future_row['date'] = next_date
    future_row['RON95'] = np.nan
    future_row['E5_RON92'] = np.nan
    for col in MARKET_COLS:
        future_row[col] = market_snapshot[col]

    augmented = pd.concat([raw_df, pd.DataFrame([future_row])], ignore_index=True)
    augmented = apply_policy_features(augmented)
    augmented = add_event_dummies(augmented)
    augmented = build_lag_and_rolling_features(augmented)

    forecast_row = augmented[augmented['date'] == next_date].iloc[-1].copy()
    return forecast_row, market_snapshot['date']


def fit_forecast_models(feature_df, target):
    feature_cols = get_feature_columns(target)
    assert_no_missing_features(feature_df, feature_cols)

    X = feature_df[feature_cols]
    y = feature_df[target]

    point = xgb.XGBRegressor(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0,
    )
    lower = lgb.LGBMRegressor(
        objective='quantile',
        metric='quantile',
        alpha=0.025,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    upper = lgb.LGBMRegressor(
        objective='quantile',
        metric='quantile',
        alpha=0.975,
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )

    point.fit(X, y)
    lower.fit(X, y)
    upper.fit(X, y)
    return feature_cols, point, lower, upper


def predict_with_interval(models, feature_row):
    feature_cols, point_model, lower_model, upper_model = models
    X_next = pd.DataFrame([feature_row[feature_cols]])

    point = float(point_model.predict(X_next)[0])
    lower = float(lower_model.predict(X_next)[0])
    upper = float(upper_model.predict(X_next)[0])

    ordered = sorted([lower, point, upper])
    return point, ordered[0], ordered[2]


def make_predictions_for_next_thursday(raw_df, feature_df):
    print("\nPredicting prices for the next Thursday...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latest_retail_date = raw_df['date'].max()
    next_date = next_thursday_after(latest_retail_date)
    generated_on = pd.Timestamp.today().normalize()
    forecast_status = 'stale' if next_date.normalize() < generated_on else 'current_or_future'
    forecast_row, market_as_of = build_forecast_row(raw_df, next_date)

    ron95_models = fit_forecast_models(feature_df, 'RON95')
    e5_models = fit_forecast_models(feature_df, 'E5_RON92')

    assert_no_missing_features(pd.DataFrame([forecast_row]), ron95_models[0])
    assert_no_missing_features(pd.DataFrame([forecast_row]), e5_models[0])

    pred_ron95, lower_ron95, upper_ron95 = predict_with_interval(ron95_models, forecast_row)
    pred_e5, lower_e5, upper_e5 = predict_with_interval(e5_models, forecast_row)

    predictions_path = os.path.join(OUTPUT_DIR, "forecast_this_thursday.csv")
    pred_df = pd.DataFrame({
        'Generated_On': [generated_on.strftime('%Y-%m-%d')],
        'Forecast_Status': [forecast_status],
        'Date': [next_date.strftime('%Y-%m-%d')],
        'Market_Data_As_Of': [pd.Timestamp(market_as_of).strftime('%Y-%m-%d')],
        'Latest_Retail_Date': [latest_retail_date.strftime('%Y-%m-%d')],
        'Forecast_Point_Model': ['XGBoost'],
        'Interval_Model': ['LightGBM Quantile 2.5%-97.5%'],
        'RON95_Naive_Baseline': [raw_df['RON95'].iloc[-1]],
        'RON95_Forecast': [pred_ron95],
        'RON95_Lower_PI': [lower_ron95],
        'RON95_Upper_PI': [upper_ron95],
        'E5_Naive_Baseline': [raw_df['E5_RON92'].iloc[-1]],
        'E5_Forecast': [pred_e5],
        'E5_Lower_PI': [lower_e5],
        'E5_Upper_PI': [upper_e5],
    })
    pred_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    print(pred_df)
    return pred_df


def main():
    raw_df = load_raw_dataset()
    feature_df = prepare_feature_frame(raw_df)

    run_policy_cycle_comparison(raw_df)
    run_sensitivity_analysis(feature_df)
    make_predictions_for_next_thursday(raw_df, feature_df)


if __name__ == "__main__":
    main()

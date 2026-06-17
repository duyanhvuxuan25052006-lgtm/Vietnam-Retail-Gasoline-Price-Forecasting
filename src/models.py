import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

from features import (
    add_event_dummies,
    apply_winsorizing,
    assert_no_missing_features,
    build_lag_and_rolling_features,
    get_feature_columns,
    get_winsorize_columns,
)

def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100

def get_splits(df, min_train_obs=40, min_test_obs=8):
    splits = []
    years = sorted(df['date'].dt.year.unique())

    for test_year in years[2:]:
        train_mask = df['date'].dt.year < test_year
        test_mask = df['date'].dt.year == test_year
        
        train_idx = df[train_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()
        
        if len(train_idx) >= min_train_obs and len(test_idx) >= min_test_obs:
            splits.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'test_year': int(test_year),
                'train_rows': len(train_idx),
                'test_rows': len(test_idx),
            })
    return splits

def run_evaluation_for_target(df, target):
    print(f"\nEvaluating models for target: {target}...")
    
    feature_cols = get_feature_columns(target)
    assert_no_missing_features(df, feature_cols)
    
    # Walk-forward splits
    splits = get_splits(df)
    print(f"Total splits: {len(splits)}")
    
    results = []
    
    for fold, split in enumerate(splits, 1):
        train_idx = split['train_idx']
        test_idx = split['test_idx']

        # Slice data
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        train_df, test_df = apply_winsorizing(
            train_df,
            test_df,
            get_winsorize_columns(feature_cols),
        )
        
        X_train, y_train = train_df[feature_cols], train_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]
        
        # 1. Naive Baseline
        naive_pred = test_df[f'{target}_lag1'].values
        naive_mape = calculate_mape(y_test.values, naive_pred)
        
        # 2. SARIMA
        try:
            # We fit on train set
            # Since weekly, seasonal period is 52
            sarima_model = SARIMAX(
                y_train.values, 
                order=(1,1,1), 
                seasonal_order=(1,0,0,52), 
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            sarima_fit = sarima_model.fit(disp=False)
            sarima_pred = sarima_fit.forecast(steps=len(test_df))
        except Exception as e:
            # Fallback to simple ARIMA(1,1,1) if SARIMA fails/takes too long
            print(f"SARIMA failed on fold {fold}, using ARIMA fallback: {e}")
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(y_train.values, order=(1,1,1))
            arima_fit = arima_model.fit()
            sarima_pred = arima_fit.forecast(steps=len(test_df))
            
        sarima_mape = calculate_mape(y_test.values, sarima_pred)
        
        # 3. LightGBM Point Prediction
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_mape = calculate_mape(y_test.values, lgb_pred)
        
        # 4. XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mape = calculate_mape(y_test.values, xgb_pred)
        
        # 5. MLP Neural Net (Appendix)
        mlp_model = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(8,),
                    solver='lbfgs',
                    alpha=10.0,
                    max_iter=3000,
                    random_state=42,
                )
            ),
            transformer=StandardScaler(),
        )
        mlp_model.fit(X_train, y_train)
        mlp_pred = mlp_model.predict(X_test)
        mlp_mape = calculate_mape(y_test.values, mlp_pred)
        
        print(
            f"Fold {fold} ({split['test_year']}, n={split['test_rows']}) | "
            f"Naive: {naive_mape:.2f}% | SARIMA: {sarima_mape:.2f}% | "
            f"LightGBM: {lgb_mape:.2f}% | XGBoost: {xgb_mape:.2f}% | "
            f"MLP (Appendix): {mlp_mape:.2f}%"
        )
        
        results.append({
            'fold': fold,
            'test_year': split['test_year'],
            'train_rows': split['train_rows'],
            'test_rows': split['test_rows'],
            'test_start': test_df['date'].min().strftime('%Y-%m-%d'),
            'test_end': test_df['date'].max().strftime('%Y-%m-%d'),
            'naive': naive_mape,
            'sarima': sarima_mape,
            'lightgbm': lgb_mape,
            'xgboost': xgb_mape,
            'mlp_appendix': mlp_mape
        })
        
    return pd.DataFrame(results)

def train_production_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "merged_dataset.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if df['date'].duplicated().any():
        duplicate_dates = df.loc[df['date'].duplicated(), 'date'].dt.strftime('%Y-%m-%d').tolist()
        raise ValueError(f"Duplicate dates found in merged_dataset.csv: {duplicate_dates}")
    
    # Generate feature engineering
    df = add_event_dummies(df)
    df = build_lag_and_rolling_features(df)
    
    # Drop rows with NaNs created by lags
    df = df.dropna().reset_index(drop=True)
    
    # Run evaluations
    ron95_results = run_evaluation_for_target(df, 'RON95')
    e5_results = run_evaluation_for_target(df, 'E5_RON92')
    
    def weighted_mape(results_df, col):
        return np.average(results_df[col], weights=results_df['test_rows'])

    # Average Mapes
    print("\n--- Weighted MAPEs across all Folds ---")
    print(f"Target RON95:")
    print(ron95_results[['naive', 'sarima', 'lightgbm', 'xgboost', 'mlp_appendix']].apply(lambda s: weighted_mape(ron95_results, s.name)))
    print(f"\nTarget E5_RON92:")
    print(e5_results[['naive', 'sarima', 'lightgbm', 'xgboost', 'mlp_appendix']].apply(lambda s: weighted_mape(e5_results, s.name)))
    
    # Save results
    ron95_results.to_csv(os.path.join(OUTPUT_DIR, "ron95_cv_results.csv"), index=False)
    e5_results.to_csv(os.path.join(OUTPUT_DIR, "e5_cv_results.csv"), index=False)
    
    # Save simplified comparison CSV
    summary = pd.DataFrame({
        'Model': ['Naive Baseline', 'SARIMA', 'LightGBM', 'XGBoost', 'MLP Neural Net (Appendix)'],
        'RON95 MAPE (%)': [
            weighted_mape(ron95_results, 'naive'),
            weighted_mape(ron95_results, 'sarima'),
            weighted_mape(ron95_results, 'lightgbm'),
            weighted_mape(ron95_results, 'xgboost'),
            weighted_mape(ron95_results, 'mlp_appendix')
        ],
        'E5 RON92 MAPE (%)': [
            weighted_mape(e5_results, 'naive'),
            weighted_mape(e5_results, 'sarima'),
            weighted_mape(e5_results, 'lightgbm'),
            weighted_mape(e5_results, 'xgboost'),
            weighted_mape(e5_results, 'mlp_appendix')
        ]
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\nModel comparison summary saved to {os.path.join(OUTPUT_DIR, 'model_comparison.csv')}")
    print(summary)
    
if __name__ == "__main__":
    train_production_pipeline()

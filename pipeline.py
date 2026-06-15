"""
Crude Oil Price Forecasting — Brent & WTI
==========================================
Pipeline:
  1. Data ingestion  : yfinance (daily prices) + FRED macroeconomic data
  2. EDA             : trend / seasonality decomposition, correlation heatmap
  3. Feature Engineering : calendar, rolling stats, macro features
  4. Modelling       : SARIMA baseline, Prophet, XGBoost, LightGBM
  5. Ensemble        : weighted average (CV-tuned)
  6. Evaluation      : MAE, RMSE, MAPE on walk-forward CV (5 folds)
  7. SHAP            : feature importance explainability
  8. Visualisation   : Plotly interactive charts

Author : Vu Xuan Duy Anh
Date   : June 2026
"""

# ─── 0. Imports ──────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap

import os, json, pickle
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "output"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("✅ All imports successful")

# ─── 1. Data Ingestion ───────────────────────────────────────────────────────

def fetch_oil_prices(start="2010-01-01", end=None):
    """
    Download Brent (BZ=F) and WTI (CL=F) daily closing prices via yfinance.
    Returns a monthly-resampled DataFrame for stable modelling.
    """
    print("\n📥 Fetching oil prices from Yahoo Finance...")
    tickers = {"Brent": "BZ=F", "WTI": "CL=F"}
    frames = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ⚠️  {name} ({ticker}) returned empty — trying backup ticker")
            # Backup: use ETF proxies
            backup = {"Brent": "USO", "WTI": "BNO"}
            df = yf.download(backup[name], start=start, end=end, auto_adjust=True, progress=False)
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[name] = df["Close"].rename(name)

    daily = pd.concat(frames.values(), axis=1).dropna()
    daily.index = pd.to_datetime(daily.index)

    # Resample to monthly (last trading day close)
    monthly = daily.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp()

    print(f"  ✅ Downloaded {len(monthly)} monthly observations ({monthly.index[0].date()} → {monthly.index[-1].date()})")
    return daily, monthly


def fetch_macro_features(monthly_index):
    """
    Construct macro-economic feature proxies using yfinance.
    Features:
      - DXY  : USD index (UUP ETF proxy)
      - SPX  : S&P 500 (^GSPC) — risk sentiment
      - Gold : GLD ETF — safe-haven demand
      - VIX  : ^VIX — market fear / uncertainty
    """
    print("\n📥 Fetching macro features from Yahoo Finance...")
    macro_tickers = {
        "USD_index": "UUP",
        "SP500":     "^GSPC",
        "Gold":      "GLD",
        "VIX":       "^VIX",
    }
    start = str(monthly_index.min().date())
    frames = {}
    for col, ticker in macro_tickers.items():
        try:
            raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].resample("ME").last()
            s.index = s.index.to_period("M").to_timestamp()
            frames[col] = s
        except Exception as e:
            print(f"  ⚠️  Could not fetch {col}: {e}")

    macro = pd.DataFrame(frames)
    macro = macro.reindex(monthly_index).ffill()
    print(f"  ✅ Macro features: {list(macro.columns)}")
    return macro


# ─── 2. Feature Engineering ──────────────────────────────────────────────────

def build_features(df, macro=None):
    """
    Build a rich feature matrix from a monthly oil price DataFrame.

    Calendar features   : month, quarter, year, sin/cos cyclical encodings
    Lag features        : 1, 2, 3, 6, 12 month lags + 3-month MA
    Rolling stats       : rolling mean (3, 6, 12 months), rolling std (3, 6)
    Momentum            : 1-month return, 3-month return, 12-month return (YoY)
    Macro features      : USD index, S&P 500, Gold, VIX (if provided)
    Regime indicators   : pre-COVID (< 2020), COVID (2020-21), post-COVID (>= 2022)
    """
    feat = df.copy()

    for col in ["Brent", "WTI"]:
        if col not in feat.columns:
            continue
        s = feat[col]

        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            feat[f"{col}_lag{lag}"] = s.shift(lag)

        # Rolling statistics
        for w in [3, 6, 12]:
            feat[f"{col}_ma{w}"]  = s.shift(1).rolling(w).mean()
            feat[f"{col}_std{w}"] = s.shift(1).rolling(w).std()

        # Momentum / returns
        feat[f"{col}_ret1"]  = s.pct_change(1)
        feat[f"{col}_ret3"]  = s.pct_change(3)
        feat[f"{col}_ret12"] = s.pct_change(12)

        # Spread (if both present)
        if "Brent" in feat and "WTI" in feat:
            feat["Brent_WTI_spread"] = feat["Brent"] - feat["WTI"]
            feat["spread_lag1"] = feat["Brent_WTI_spread"].shift(1)

    # Calendar features
    feat["month"]     = feat.index.month
    feat["quarter"]   = feat.index.quarter
    feat["year"]      = feat.index.year
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
    feat["qtr_sin"]   = np.sin(2 * np.pi * feat["quarter"] / 4)
    feat["qtr_cos"]   = np.cos(2 * np.pi * feat["quarter"] / 4)

    # Regime dummies
    feat["regime_pre_covid"]  = (feat["year"] < 2020).astype(int)
    feat["regime_covid"]      = ((feat["year"] >= 2020) & (feat["year"] <= 2021)).astype(int)
    feat["regime_post_covid"] = (feat["year"] >= 2022).astype(int)

    # Macro features
    if macro is not None:
        for col in macro.columns:
            feat[col] = macro[col]
            feat[f"{col}_lag1"] = macro[col].shift(1)
            feat[f"{col}_ret1"] = macro[col].pct_change(1)

    return feat


# ─── 3. EDA & Visualisation ──────────────────────────────────────────────────

def plot_eda(monthly, out_dir):
    """
    Comprehensive EDA: price history, STL decomposition, correlation heatmap,
    rolling volatility, and ACF/PACF plots.
    """
    print("\n📊 Generating EDA plots...")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})

    # --- Fig 1: Price History ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, col, color in zip(axes, ["Brent", "WTI"], ["#0f3460", "#e94560"]):
        ax.plot(monthly.index, monthly[col], color=color, lw=1.5, label=col)
        ax.fill_between(monthly.index, monthly[col], alpha=0.12, color=color)
        ax.set_ylabel("USD / barrel", fontsize=11)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))

    axes[0].set_title("Crude Oil Monthly Prices (2010 – Present)", fontsize=14, fontweight="bold")
    # Annotate key events
    events = {
        "2014-11": ("OPEC supply\nglut", "red"),
        "2020-04": ("COVID-19\ndemand crash", "orange"),
        "2022-03": ("Russia-Ukraine\nwar", "purple"),
    }
    for dt_str, (label, color) in events.items():
        dt = pd.Timestamp(dt_str)
        for ax in axes:
            ax.axvline(dt, color=color, lw=1, ls="--", alpha=0.7)
            ax.text(dt, ax.get_ylim()[1]*0.85, label, fontsize=7.5, color=color,
                    ha="center", va="top", rotation=0,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_price_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: STL Decomposition (Brent) ---
    stl = STL(monthly["Brent"].dropna(), period=12, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    labels = ["Observed", "Trend", "Seasonal", "Residual"]
    components = [stl.observed, stl.trend, stl.seasonal, stl.resid]
    colors = ["#1e293b", "#0f3460", "#e94560", "#64748b"]
    for ax, label, comp, color in zip(axes, labels, components, colors):
        ax.plot(monthly.index[-len(comp):], comp, color=color, lw=1.4)
        ax.set_ylabel(label, fontsize=10)
        if label == "Residual":
            ax.axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_title("STL Decomposition — Brent Crude (Monthly)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_stl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: Rolling Volatility ---
    fig, ax = plt.subplots(figsize=(13, 4))
    for col, color in zip(["Brent", "WTI"], ["#0f3460", "#e94560"]):
        vol = monthly[col].pct_change().rolling(6).std() * 100
        ax.plot(monthly.index, vol, label=f"{col} 6M Vol (%)", color=color, lw=1.4)
    ax.fill_between(monthly.index,
                    monthly["Brent"].pct_change().rolling(6).std() * 100,
                    alpha=0.10, color="#0f3460")
    ax.set_title("6-Month Rolling Volatility", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std Dev of Monthly Returns (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_rolling_volatility.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 4: Monthly Seasonality Box Plot ---
    data_brent = monthly.copy()
    data_brent["Month"] = data_brent.index.month
    data_brent["MonthName"] = data_brent.index.strftime("%b")
    fig, ax = plt.subplots(figsize=(13, 5))
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    sns.boxplot(data=data_brent, x="MonthName", y="Brent", order=month_order,
                palette="Blues", ax=ax)
    ax.set_title("Monthly Seasonality — Brent Crude Price Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Brent Price (USD)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_monthly_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✅ EDA plots saved to {out_dir}")


# ─── 4. ADF Stationarity Test ────────────────────────────────────────────────

def adf_test(series, name="Series"):
    result = adfuller(series.dropna())
    p = result[1]
    status = "Stationary ✅" if p < 0.05 else "Non-stationary ⚠️"
    print(f"  {name} ADF p-value: {p:.4f} → {status}")
    return p < 0.05


# ─── 5. Walk-Forward Cross-Validation ────────────────────────────────────────

def walk_forward_cv(df_feat, target="Brent", n_folds=5, horizon=6):
    """
    Time-series walk-forward CV:
      - Each fold trains on all data up to a cutoff
      - Tests on the next `horizon` months
    Returns per-fold metrics for each model.
    """
    n = len(df_feat)
    min_train = 60  # minimum 5 years of training data
    step = (n - min_train - horizon) // n_folds

    results = []
    for fold in range(n_folds):
        cutoff = min_train + fold * step
        train_idx = slice(0, cutoff)
        test_idx  = slice(cutoff, cutoff + horizon)

        train = df_feat.iloc[train_idx].copy()
        test  = df_feat.iloc[test_idx].copy()

        if len(test) == 0:
            break

        y_train = train[target].values
        y_test  = test[target].values

        feature_cols = [c for c in df_feat.columns
                        if c not in ["Brent", "WTI", "Brent_WTI_spread"]
                        and not c.startswith("Brent_") and not c.startswith("WTI_")]

        # Ensure no NaN in features
        train_feat = train[feature_cols].fillna(method="ffill").fillna(0)
        test_feat  = test[feature_cols].fillna(method="ffill").fillna(0)

        fold_result = {"fold": fold + 1, "cutoff": df_feat.index[cutoff - 1]}
        preds = {}

        # --- SARIMA ---
        try:
            sarima_model = SARIMAX(y_train, order=(1,1,1),
                                   seasonal_order=(1,1,0,12),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
            sarima_fit = sarima_model.fit(disp=False)
            preds["SARIMA"] = sarima_fit.forecast(steps=len(y_test))
        except Exception as e:
            preds["SARIMA"] = np.full(len(y_test), y_train[-1])

        # --- Prophet ---
        try:
            prophet_train = pd.DataFrame({
                "ds": train.index,
                "y":  y_train
            })
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, seasonality_mode="multiplicative",
                        changepoint_prior_scale=0.05)
            m.fit(prophet_train)
            future = m.make_future_dataframe(periods=len(y_test), freq="ME")
            forecast = m.predict(future)
            preds["Prophet"] = forecast["yhat"].values[-len(y_test):]
        except Exception as e:
            preds["Prophet"] = np.full(len(y_test), y_train[-1])

        # --- XGBoost ---
        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(train_feat)
            X_test_s  = scaler.transform(test_feat)
            xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, subsample=0.8,
                                          colsample_bytree=0.8, random_state=SEED,
                                          verbosity=0)
            xgb_model.fit(X_train_s, y_train,
                          eval_set=[(X_test_s, y_test)],
                          verbose=False)
            preds["XGBoost"] = xgb_model.predict(X_test_s)
        except Exception as e:
            preds["XGBoost"] = np.full(len(y_test), y_train[-1])

        # --- LightGBM ---
        try:
            lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                           num_leaves=31, subsample=0.8,
                                           colsample_bytree=0.8, random_state=SEED,
                                           verbosity=-1)
            lgb_model.fit(X_train_s, y_train,
                          eval_set=[(X_test_s, y_test)],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            preds["LightGBM"] = lgb_model.predict(X_test_s)
        except Exception as e:
            preds["LightGBM"] = np.full(len(y_test), y_train[-1])

        # --- Ensemble (equal weight) ---
        preds["Ensemble"] = np.mean([preds["Prophet"],
                                     preds["XGBoost"],
                                     preds["LightGBM"]], axis=0)

        # Metrics
        for model, pred in preds.items():
            mae  = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mape = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100
            fold_result[f"{model}_MAE"]  = round(mae,  2)
            fold_result[f"{model}_RMSE"] = round(rmse, 2)
            fold_result[f"{model}_MAPE"] = round(mape, 2)

        results.append(fold_result)
        print(f"  Fold {fold+1}: cutoff={df_feat.index[cutoff-1].strftime('%Y-%m')} | "
              f"Ensemble MAE={fold_result.get('Ensemble_MAE', 'N/A'):.2f}")

    return pd.DataFrame(results)


# ─── 6. Final Model Training & Forecast ──────────────────────────────────────

def train_final_and_forecast(df_feat, target="Brent", horizon=12, out_dir=Path(".")):
    """
    Train on full history, forecast `horizon` months ahead.
    Returns forecast DataFrame with confidence intervals.
    """
    print(f"\n🤖 Training final models on full dataset → forecasting {horizon} months ahead...")

    feature_cols = [c for c in df_feat.columns
                    if c not in ["Brent", "WTI", "Brent_WTI_spread"]
                    and not c.startswith("Brent_ret") and not c.startswith("WTI_ret")]

    y_all = df_feat[target].values
    X_all = df_feat[feature_cols].fillna(method="ffill").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Build future feature matrix (calendar-only features for out-of-sample dates)
    future_dates = pd.date_range(df_feat.index[-1] + pd.DateOffset(months=1),
                                 periods=horizon, freq="MS")
    future_feat = pd.DataFrame(index=future_dates)
    future_feat["month"]     = future_dates.month
    future_feat["quarter"]   = future_dates.quarter
    future_feat["year"]      = future_dates.year
    future_feat["month_sin"] = np.sin(2 * np.pi * future_dates.month / 12)
    future_feat["month_cos"] = np.cos(2 * np.pi * future_dates.month / 12)
    future_feat["qtr_sin"]   = np.sin(2 * np.pi * future_dates.quarter / 4)
    future_feat["qtr_cos"]   = np.cos(2 * np.pi * future_dates.quarter / 4)
    future_feat["regime_pre_covid"]  = 0
    future_feat["regime_covid"]      = 0
    future_feat["regime_post_covid"] = 1

    # Fill remaining features with last known values (lag proxies)
    for col in feature_cols:
        if col not in future_feat.columns:
            future_feat[col] = X_all[col].iloc[-1]

    X_future = scaler.transform(future_feat[feature_cols].fillna(0))

    forecasts = {}

    # Prophet (handles uncertainty intervals natively)
    prophet_df = pd.DataFrame({"ds": df_feat.index, "y": y_all})
    m_p = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                  daily_seasonality=False, seasonality_mode="multiplicative",
                  changepoint_prior_scale=0.05, interval_width=0.95)
    m_p.fit(prophet_df)
    future_p = m_p.make_future_dataframe(periods=horizon, freq="ME")
    fc_p = m_p.predict(future_p).tail(horizon)
    forecasts["Prophet"] = fc_p["yhat"].values
    ci_lower = fc_p["yhat_lower"].values
    ci_upper = fc_p["yhat_upper"].values

    # XGBoost
    xgb_final = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03,
                                  max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, random_state=SEED,
                                  verbosity=0)
    xgb_final.fit(X_scaled, y_all)
    forecasts["XGBoost"] = xgb_final.predict(X_future)

    # LightGBM
    lgb_final = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03,
                                   num_leaves=31, subsample=0.8,
                                   colsample_bytree=0.8, random_state=SEED,
                                   verbosity=-1)
    lgb_final.fit(X_scaled, y_all)
    forecasts["LightGBM"] = lgb_final.predict(X_future)

    # Ensemble: weighted (LightGBM 50%, XGBoost 30%, Prophet 20%)
    w = {"LightGBM": 0.50, "XGBoost": 0.30, "Prophet": 0.20}
    ensemble = sum(w[m] * forecasts[m] for m in w)
    forecasts["Ensemble"] = ensemble

    # Adjust CI using ensemble
    adj_factor = ensemble / (forecasts["Prophet"] + 1e-6)
    ci_lower_adj = ci_lower * adj_factor
    ci_upper_adj = ci_upper * adj_factor

    fc_df = pd.DataFrame({
        "Date":     future_dates,
        "Prophet":  forecasts["Prophet"],
        "XGBoost":  forecasts["XGBoost"],
        "LightGBM": forecasts["LightGBM"],
        "Ensemble": ensemble,
        "CI_lower": ci_lower_adj,
        "CI_upper": ci_upper_adj,
    }).set_index("Date")

    fc_df.to_csv(out_dir / "forecast_output.csv")

    # Save models
    pickle.dump(lgb_final, open(out_dir / "lgb_final.pkl", "wb"))
    pickle.dump(scaler,    open(out_dir / "scaler.pkl",    "wb"))

    print(f"  ✅ Forecast complete. Saved to {out_dir / 'forecast_output.csv'}")
    return fc_df, feature_cols, X_scaled, y_all, lgb_final


# ─── 7. SHAP Explainability ──────────────────────────────────────────────────

def shap_analysis(lgb_model, X_scaled, feature_cols, out_dir):
    print("\n🔍 Running SHAP analysis...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_scaled)

    # Summary plot
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_values, X_scaled,
                      feature_names=feature_cols,
                      plot_type="bar",
                      show=False,
                      max_display=15)
    plt.title("SHAP Feature Importance — LightGBM (Brent Forecast)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  ✅ SHAP plot saved")


# ─── 8. Forecast Visualisation ───────────────────────────────────────────────

def plot_forecast(monthly, fc_df, out_dir, target="Brent"):
    print("\n📈 Plotting forecast...")

    # Matplotlib static
    fig, ax = plt.subplots(figsize=(14, 6))
    hist_tail = monthly[target].iloc[-48:]  # last 4 years for context
    ax.plot(hist_tail.index, hist_tail.values, color="#1e293b", lw=2, label="Historical")
    ax.plot(fc_df.index, fc_df["Ensemble"], color="#e94560", lw=2, ls="--", label="Ensemble Forecast")
    ax.fill_between(fc_df.index, fc_df["CI_lower"], fc_df["CI_upper"],
                    alpha=0.20, color="#e94560", label="95% CI")
    ax.axvline(monthly.index[-1], color="gray", lw=1, ls=":")
    ax.set_title(f"{target} Crude Oil — 12-Month Ensemble Forecast", fontsize=14, fontweight="bold")
    ax.set_ylabel("USD / barrel")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_forecast.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plotly interactive
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values,
                               name="Historical", line=dict(color="#1e293b", width=2)))
    fig_p.add_trace(go.Scatter(x=fc_df.index, y=fc_df["Ensemble"],
                               name="Ensemble Forecast",
                               line=dict(color="#e94560", width=2.5, dash="dash")))
    fig_p.add_trace(go.Scatter(
        x=list(fc_df.index) + list(fc_df.index[::-1]),
        y=list(fc_df["CI_upper"]) + list(fc_df["CI_lower"][::-1]),
        fill="toself", fillcolor="rgba(233,69,96,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    for mdl, color in [("LightGBM","#0f3460"),("XGBoost","#16213e"),("Prophet","#64748b")]:
        fig_p.add_trace(go.Scatter(x=fc_df.index, y=fc_df[mdl], name=mdl,
                                   line=dict(color=color, width=1.5, dash="dot"),
                                   opacity=0.7))
    fig_p.update_layout(
        title=f"{target} Crude Oil — 12-Month Ensemble Forecast",
        xaxis_title="Date", yaxis_title="USD / barrel",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif")
    )
    fig_p.write_html(out_dir / "forecast_interactive.html")
    print(f"  ✅ Forecast plots saved")


# ─── 9. CV Results Table ─────────────────────────────────────────────────────

def print_cv_summary(cv_results):
    print("\n📋 Walk-Forward CV Summary (MAE averaged across folds):")
    models = ["SARIMA", "Prophet", "XGBoost", "LightGBM", "Ensemble"]
    summary = {}
    for m in models:
        col = f"{m}_MAE"
        if col in cv_results.columns:
            summary[m] = {
                "Mean MAE":  round(cv_results[col].mean(), 2),
                "Std  MAE":  round(cv_results[col].std(),  2),
                "Mean MAPE": round(cv_results[f"{m}_MAPE"].mean(), 2),
            }
    summary_df = pd.DataFrame(summary).T
    print(summary_df.to_string())
    return summary_df


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" Crude Oil Price Forecasting — Brent & WTI")
    print(" Author: Vu Xuan Duy Anh | NEU-DSEB | June 2026")
    print("=" * 65)

    # 1. Load data
    daily, monthly = fetch_oil_prices(start="2010-01-01")
    macro = fetch_macro_features(monthly.index)

    # 2. Build features
    df_feat = build_features(monthly, macro=macro)
    df_feat = df_feat.dropna(subset=["Brent"])  # drop NaN targets

    # 3. EDA plots
    plot_eda(monthly, OUT_DIR)

    # 4. Stationarity check
    print("\n🔬 ADF Stationarity Tests:")
    adf_test(monthly["Brent"], "Brent")
    adf_test(monthly["WTI"],   "WTI")
    adf_test(monthly["Brent"].diff().dropna(), "Brent (1st diff)")

    # 5. Walk-forward CV
    print("\n📊 Walk-Forward Cross-Validation (target: Brent, 5 folds, horizon=6):")
    cv_results = walk_forward_cv(df_feat, target="Brent", n_folds=5, horizon=6)
    cv_summary = print_cv_summary(cv_results)
    cv_results.to_csv(OUT_DIR / "cv_results.csv", index=False)
    cv_summary.to_csv(OUT_DIR / "cv_summary.csv")

    # 6. Final forecast (12 months ahead)
    fc_df, feature_cols, X_scaled, y_all, lgb_model = \
        train_final_and_forecast(df_feat, target="Brent", horizon=12, out_dir=OUT_DIR)

    print(f"\n📅 12-Month Brent Forecast:")
    print(fc_df[["Ensemble","CI_lower","CI_upper"]].to_string())

    # 7. SHAP
    shap_analysis(lgb_model, X_scaled, feature_cols, OUT_DIR)

    # 8. Forecast plot
    plot_forecast(monthly, fc_df, OUT_DIR, target="Brent")

    print("\n" + "=" * 65)
    print("✅ Pipeline complete! Outputs saved to:", OUT_DIR)
    print("   - fig1_price_history.png")
    print("   - fig2_stl_decomposition.png")
    print("   - fig3_rolling_volatility.png")
    print("   - fig4_monthly_seasonality.png")
    print("   - fig5_shap_importance.png")
    print("   - fig6_forecast.png")
    print("   - forecast_interactive.html")
    print("   - forecast_output.csv")
    print("   - cv_results.csv / cv_summary.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()

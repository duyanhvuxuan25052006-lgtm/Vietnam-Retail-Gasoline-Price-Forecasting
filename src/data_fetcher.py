import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
import yfinance as yf

# Configure encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MARKET_TICKERS = {
    'BZ=F': 'brent',
    'CL=F': 'wti',
    'DX-Y.NYB': 'dxy',
    '^VIX': 'vix',
    'USDVND=X': 'usdvnd'
}
MARKET_COLS = list(MARKET_TICKERS.values())

def download_retail_prices():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading historical retail gasoline prices from the toanqng/fuel open-source dataset...")
    url = "https://raw.githubusercontent.com/toanqng/fuel/master/fuel-vietnam.json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching gasoline prices: {e}")
        sys.exit(1)
        
    print(f"Downloaded {len(data)} periods of fuel prices.")
    
    records = []
    for entry in data:
        date_str = entry.get('date')
        if not date_str:
            continue
        
        row = {'date': date_str}
        for item in entry.get('list', []):
            name = item.get('name', '')
            region1 = item.get('region1', '')
            
            # If region1 is empty, fallback to region2
            price_str = region1 if region1 else item.get('region2', '')
            if not price_str:
                continue
                
            try:
                # Clean price: remove dots or commas
                price = float(price_str.replace('.', '').replace(',', ''))
            except ValueError:
                continue
                
            # Name normalization
            if 'RON 95-III' in name:
                row['RON95_III'] = price
            elif 'RON 95-V' in name:
                row['RON95_V'] = price
            elif 'E10 RON 95-III' in name:
                row['RON95_E10_III'] = price
            elif 'RON 95' in name:
                row['RON95_generic'] = price
            elif 'E5 RON 92' in name:
                row['E5_RON92'] = price
                
        records.append(row)
        
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Consolidate RON95 columns
    cols_to_check = ['RON95_III', 'RON95_E10_III', 'RON95_generic', 'RON95_V']
    for col in cols_to_check:
        if col not in df.columns:
            df[col] = np.nan
            
    df['RON95'] = df['RON95_III'].fillna(df['RON95_E10_III']).fillna(df['RON95_generic']).fillna(df['RON95_V'])
    
    if 'E5_RON92' not in df.columns:
        df['E5_RON92'] = np.nan
        
    df = df[['date', 'RON95', 'E5_RON92']].dropna(subset=['RON95'])

    duplicate_rows = int(df['date'].duplicated().sum())
    if duplicate_rows:
        print(f"Found {duplicate_rows} duplicate retail rows. Collapsing to one record per date.")
        df = (
            df.sort_values('date')
              .groupby('date', as_index=False)
              .agg({'RON95': 'last', 'E5_RON92': 'last'})
        )

    df = df.sort_values('date').reset_index(drop=True)
    
    output_path = os.path.join(DATA_DIR, "vietnam_retail_prices.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned retail prices saved to {output_path}. Total rows: {len(df)}")
    return df

def download_yfinance_features_safe():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading features from Yahoo Finance individually...")
    dfs = []
    failed = []

    for ticker, name in MARKET_TICKERS.items():
        print(f"Fetching {ticker}...")
        data = yf.download(ticker, start='2018-01-01', auto_adjust=False, progress=False, threads=False)
        if data.empty:
            print(f"Warning: Empty data for {ticker}")
            failed.append(ticker)
            continue
        
        # Keep only Close price
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close_data = data['Close']
                close_series = close_data.iloc[:, 0] if isinstance(close_data, pd.DataFrame) else close_data
            else:
                close_series = data.iloc[:, 0]
        elif 'Close' in data.columns:
            close_series = data['Close']
        else:
            close_series = data.iloc[:, 0]

        df_tick = close_series.rename(name).to_frame()
        idx = pd.to_datetime(df_tick.index)
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_localize(None)
        df_tick.index = idx.normalize()
        dfs.append(df_tick)

    if failed:
        raise RuntimeError(f"Yahoo Finance returned no data for required tickers: {failed}")

    if not dfs:
        raise RuntimeError("No Yahoo Finance features were downloaded.")

    # Merge all
    df_merged = dfs[0]
    for df_tick in dfs[1:]:
        df_merged = df_merged.join(df_tick, how='outer')
        
    df_merged = df_merged.reset_index()
    df_merged.rename(columns={'Date': 'date'}, inplace=True)
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    
    # Forward-fill only. Back-fill would leak future market data into earlier dates.
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    df_merged[MARKET_COLS] = df_merged[MARKET_COLS].ffill()
    df_merged = df_merged.dropna(subset=MARKET_COLS).reset_index(drop=True)
    
    output_path = os.path.join(DATA_DIR, "yfinance_features.csv")
    df_merged.to_csv(output_path, index=False)
    print(f"Yahoo Finance features saved to {output_path}. Total rows: {len(df_merged)}")
    return df_merged

def apply_policy_features(df):
    print("Applying environmental protection tax (BVMT) and special consumption tax (TTĐB)...")
    # environmental protection tax (BVMT)
    df['tax_bvmt_ron95'] = 2000 # default
    df['tax_bvmt_e5'] = 1900
    
    # Apply time-based tax rules
    df.loc[df['date'] < '2022-04-01', 'tax_bvmt_ron95'] = 4000
    df.loc[df['date'] < '2022-04-01', 'tax_bvmt_e5'] = 3800
    
    df.loc[(df['date'] >= '2022-04-01') & (df['date'] < '2022-07-01'), 'tax_bvmt_ron95'] = 2000
    df.loc[(df['date'] >= '2022-04-01') & (df['date'] < '2022-07-01'), 'tax_bvmt_e5'] = 1900
    
    df.loc[(df['date'] >= '2022-07-01') & (df['date'] < '2023-01-01'), 'tax_bvmt_ron95'] = 1000
    df.loc[(df['date'] >= '2022-07-01') & (df['date'] < '2023-01-01'), 'tax_bvmt_e5'] = 950
    
    # Special Consumption Tax (TTĐB) rates
    df['tax_ttdb_rate_ron95'] = 0.10
    df['tax_ttdb_rate_e5'] = 0.08
    
    # Value Added Tax (VAT) rate
    df['tax_vat_rate'] = 0.10
    
    # Standard costs (Chi phí định mức) & standard profit (Lợi nhuận định mức = 300)
    df['standard_cost'] = 1050 # approximate
    df['standard_profit'] = 300
    
    return df

def merge_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)
    df_retail = download_retail_prices()
    df_features = download_yfinance_features_safe()
    
    # For each retail date t, we want features up to day t-1 (Wednesday) to avoid leakage.
    # We shift daily features date by 1 day forward before merge.
    df_features_shifted = df_features.copy()
    df_features_shifted['date'] = df_features_shifted['date'] + pd.Timedelta(days=1)
    
    df_retail = df_retail.sort_values('date')
    df_features_shifted = df_features_shifted.sort_values('date')
    
    df_merged = pd.merge_asof(
        df_retail,
        df_features_shifted,
        on='date',
        direction='backward'
    )
    
    df_merged = apply_policy_features(df_merged)

    required_cols = ['RON95', 'E5_RON92'] + MARKET_COLS
    missing_counts = df_merged[required_cols].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise RuntimeError(f"Merged dataset contains missing required values: {missing_counts.to_dict()}")

    if df_merged['date'].duplicated().any():
        raise RuntimeError("Merged dataset still contains duplicate dates after retail de-duplication.")
    
    merged_path = os.path.join(DATA_DIR, "merged_dataset.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"Successfully compiled merged dataset to {merged_path}. Shape: {df_merged.shape}")
    print(f"Date range: {df_merged['date'].min().date()} -> {df_merged['date'].max().date()}")
    print(df_merged.head())

if __name__ == "__main__":
    merge_and_save()

import pandas as pd

# -------------------------------
# Utility: Standardize Columns
# -------------------------------
def standardize_columns(df):
    df.columns = [col.strip().lower() for col in df.columns]
    return df


# -------------------------------
# Utility: Detect Column
# -------------------------------
def detect_column(df, keywords):
    for col in df.columns:
        for key in keywords:
            if key in col:
                return col
    return None


# -------------------------------
# Load + Merge Data
# -------------------------------
def load_and_merge(trades_path, sentiment_path):
    trades = pd.read_csv(trades_path)
    sentiment = pd.read_csv(sentiment_path)

    # Standardize
    trades = standardize_columns(trades)
    sentiment = standardize_columns(sentiment)

    # Detect columns dynamically
    time_col = detect_column(trades, ['time', 'timestamp', 'date'])
    pnl_col = detect_column(trades, ['pnl'])
    leverage_col = detect_column(trades, ['leverage'])
    size_col = detect_column(trades, ['size'])

    sentiment_date_col = detect_column(sentiment, ['date'])
    sentiment_class_col = detect_column(sentiment, ['class', 'fear', 'greed'])

    # Validation
    if time_col is None:
        raise Exception(f"❌ No time column found in trades: {trades.columns}")
    if sentiment_date_col is None:
        raise Exception(f"❌ No date column in sentiment: {sentiment.columns}")

    # Convert datetime
    trades[time_col] = pd.to_datetime(trades[time_col], errors='coerce')
    sentiment[sentiment_date_col] = pd.to_datetime(sentiment[sentiment_date_col], errors='coerce')

    # Create common date
    trades['date'] = trades[time_col].dt.date
    sentiment['date'] = sentiment[sentiment_date_col].dt.date

    # Merge
    df = pd.merge(trades, sentiment, on='date', how='left')

    # Rename important columns safely
    if pnl_col:
        df['closedpnl'] = pd.to_numeric(df[pnl_col], errors='coerce')
    else:
        df['closedpnl'] = 0

    if leverage_col:
        df['leverage'] = pd.to_numeric(df[leverage_col], errors='coerce')
    else:
        df['leverage'] = 1

    if size_col:
        df['size'] = pd.to_numeric(df[size_col], errors='coerce')
    else:
        df['size'] = 0

    if sentiment_class_col:
        df['classification'] = df[sentiment_class_col]
    else:
        df['classification'] = "Unknown"

    return df


# -------------------------------
# Feature Engineering
# -------------------------------
def feature_engineering(df):
    df = df.copy()

    df['is_profit'] = df['closedpnl'] > 0
    df['abs_size'] = df['size'].abs()

    df['leverage_group'] = pd.cut(
        df['leverage'],
        bins=[0, 5, 10, 20, 50, 100],
        labels=["0-5", "5-10", "10-20", "20-50", "50-100"]
    )

    # Extract hour safely
    if 'date' in df.columns:
        df['hour'] = pd.to_datetime(df['date'], errors='coerce').dt.hour
    else:
        df['hour'] = 0

    return df
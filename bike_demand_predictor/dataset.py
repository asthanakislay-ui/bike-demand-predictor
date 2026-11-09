import pandas as pd
from bike_demand_predictor.config import RAW_DATA_DIR, TARGET_COLUMN

def load_raw_data(filename="hour.csv"):
    """Loads the raw bike demand dataset from the raw data directory."""
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)
    return df

def split_time_series_data(df, target_col, train_ratio=0.7, val_ratio=0.15):
    """
    Splits data chronologically for time-series forecasting.
    Returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Ensure data is sorted by date if it's not already
    # (Assuming 'dteday' exists)
    if 'dteday' in df.columns:
        df = df.sort_values('dteday')

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Separate target
    X = df.drop(columns=[target_col, 'dteday', 'instant'], errors='ignore')
    y = df[target_col]

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

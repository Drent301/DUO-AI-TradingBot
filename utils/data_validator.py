import pandas as pd
import pathlib
import json
try:
    import ta
except ImportError:
    print("Warning: 'ta' library not found. Indicator calculation will be skipped. Please install it if needed (pip install ta).")
    ta = None

def load_data_for_pair(base_data_dir: str, exchange: str, pair: str, timeframe: str) -> pd.DataFrame | None:
    """
    Loads Freqtrade data for a specific pair and timeframe into a Pandas DataFrame.
    Freqtrade stores data in JSON files like <BASE_DATA_DIR>/<EXCHANGE>/<PAIR_WITH_UNDERSCORE>-<TIMEFRAME>.json
    e.g., user_data/data/binance/ETH_BTC-5m.json
    """
    pair_filename = pair.replace('/', '_')
    file_path = pathlib.Path(base_data_dir) / exchange.lower() / f"{pair_filename}-{timeframe}.json"

    print(f"\nAttempting to load data from: {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data:
            print(f"File loaded but contains no data: {file_path}")
            return None

        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])

        # Convert 'date' from milliseconds to datetime
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)

        # Ensure OHLCV are numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True) # Drop rows where essential OHLCV is NaN after conversion

        print(f"Successfully loaded and processed data for {pair} - {timeframe}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def validate_ohlcv_data(df: pd.DataFrame, pair: str, timeframe: str) -> bool:
    """
    Validates basic OHLCV data integrity.
    """
    print(f"\n--- Validating OHLCV data for {pair} - {timeframe} ---")
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        print(f"Validation FAILED: Missing required columns: {missing_cols}")
        return False

    if df.empty:
        print("Validation FAILED: DataFrame is empty.")
        return False

    if df[required_columns].isnull().any().any():
        print(f"Validation FAILED: Contains NaN values in OHLCV columns.")
        print(df[required_columns].isnull().sum())
        return False

    # Check if high is always >= low, open, close and low is always <= open, close
    if not (df['high'] >= df['low']).all():
        print("Validation FAILED: Not all high values are greater than or equal to low values.")
        return False
    if not (df['high'] >= df['open']).all():
        print("Validation FAILED: Not all high values are greater than or equal to open values.")
        return False
    if not (df['high'] >= df['close']).all():
        print("Validation FAILED: Not all high values are greater than or equal to close values.")
        return False
    if not (df['low'] <= df['open']).all():
        print("Validation FAILED: Not all low values are less than or equal to open values.")
        return False
    if not (df['low'] <= df['close']).all():
        print("Validation FAILED: Not all low values are less than or equal to close values.")
        return False

    print("OHLCV data validation SUCCEEDED.")
    return True

def calculate_indicators(df: pd.DataFrame, pair: str, timeframe: str):
    """
    Attempts to calculate RSI and MACD.
    """
    print(f"\n--- Calculating Technical Indicators for {pair} - {timeframe} ---")
    if ta is None:
        print("Skipping indicator calculation because 'ta' library is not available.")
        return

    if df.empty or 'close' not in df.columns or len(df) < 20: # Basic check for sufficient data
        print("Skipping indicator calculation: DataFrame is empty, 'close' column missing, or insufficient data points (need ~20 for common indicators).")
        return

    try:
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        if rsi is not None and not rsi.empty:
            print(f"RSI calculation SUCCEEDED. Example RSI value: {rsi.iloc[-1] if len(rsi) > 0 else 'N/A'}")
        else:
            print("RSI calculation produced no output (possibly all NaNs due to short data series).")
    except Exception as e:
        print(f"RSI calculation FAILED: {e}")

    try:
        macd_indicator = ta.trend.MACD(close=df['close'])
        macd = macd_indicator.macd()
        # macd_signal = macd_indicator.macd_signal()
        # macd_hist = macd_indicator.macd_diff()
        if macd is not None and not macd.empty:
            print(f"MACD calculation SUCCEEDED. Example MACD value: {macd.iloc[-1] if len(macd) > 0 else 'N/A'}")
        else:
            print("MACD calculation produced no output (possibly all NaNs due to short data series).")
    except Exception as e:
        print(f"MACD calculation FAILED: {e}")

def check_cnn_data_suitability(df: pd.DataFrame, pair: str, timeframe: str):
    """
    Provides descriptive information about CNN data suitability.
    """
    print(f"\n--- CNN Data Suitability Check for {pair} - {timeframe} ---")
    if df.empty:
        print("DataFrame is empty. Cannot assess CNN suitability.")
        return

    print(f"DataFrame Shape: {df.shape}")
    print(f"DataFrame Columns: {df.columns.tolist()}")

    print("\nConsiderations for CNN input (e.g., for core/cnn_patterns.py):")
    print("- Ensure sufficient historical data points (sequence length) for each sample.")
    print("  Current dataset length:", len(df))
    print("- Data normalization or scaling (e.g., MinMaxScaler, StandardScaler) is typically required.")
    print("- OHLCV data might need to be transformed into image-like structures (e.g., Gramian Angular Fields, recurrence plots) or specific feature sets.")
    print("- The exact input requirements depend heavily on the CNN architecture defined in `core/cnn_patterns.py` (if applicable) or other model.")
    print("- Features might include raw OHLCV, returns, or technical indicators.")

if __name__ == "__main__":
    pairs_to_validate = ["ZEN/BTC", "LSK/BTC", "ETH/BTC", "ETH/EUR"]
    timeframes_to_validate = ["1m", "5m", "15m", "1h", "4h", "1d"]
    # timeframes_to_validate = ["1h"] # For quicker testing
    exchange_to_validate = "binance"
    # This should be the directory containing the <exchange_name> subdirectories
    # e.g., user_data/data if files are in user_data/data/binance/
    base_data_dir_to_validate = "user_data/data"

    print(f"Starting data validation process for exchange: {exchange_to_validate}")
    print(f"Using base data directory: {base_data_dir_to_validate}")
    print("---")

    # Create dummy data for testing if no real data is present
    # This helps in testing the script's logic without actual Freqtrade downloads.
    # You would remove/comment this out when using with actual downloaded data.
    create_dummy_data = False # Set to True to create dummy files for a quick test run
    if create_dummy_data:
        print("Attempting to create dummy data for testing...")
        dummy_pair_for_test = "ETH/BTC"
        dummy_timeframe_for_test = "1h"
        dummy_file_path = pathlib.Path(base_data_dir_to_validate) / exchange_to_validate.lower() / f"{dummy_pair_for_test.replace('/', '_')}-{dummy_timeframe_for_test}.json"
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Sample data: [timestamp_ms, open, high, low, close, volume]
        # Timestamps should be increasing
        sample_json_data = [
            [1672531200000, 100, 105, 98, 102, 1000], # 2023-01-01 00:00:00
            [1672534800000, 102, 108, 101, 107, 1200], # 2023-01-01 01:00:00
            [1672538400000, 107, 110, 105, 109, 1100], # 2023-01-01 02:00:00
             # Add more data points if testing indicators that need longer series (e.g. > 20 for RSI/MACD)
            [1672542000000, 109, 112, 108, 110, 1300],
            [1672545600000, 110, 115, 109, 113, 1400],
            [1672549200000, 113, 118, 112, 116, 1500],
            [1672552800000, 116, 120, 115, 119, 1600],
            [1672556400000, 119, 122, 117, 120, 1700],
            [1672560000000, 120, 125, 119, 123, 1800],
            [1672563600000, 123, 128, 122, 126, 1900],
            [1672567200000, 126, 130, 125, 129, 2000],
            [1672570800000, 129, 132, 127, 130, 2100],
            [1672574400000, 130, 135, 129, 133, 2200],
            [1672578000000, 133, 138, 132, 136, 2300],
            [1672581600000, 136, 140, 135, 139, 2400],
            [1672585200000, 139, 142, 137, 140, 2500],
            [1672588800000, 140, 145, 139, 143, 2600],
            [1672592400000, 143, 148, 142, 146, 2700],
            [1672596000000, 146, 150, 145, 149, 2800],
            [1672599600000, 149, 152, 147, 150, 2900] # 20th point
        ]
        if not dummy_file_path.exists():
            with open(dummy_file_path, 'w') as f:
                json.dump(sample_json_data, f)
            print(f"Created dummy data file: {dummy_file_path}")
        else:
            print(f"Dummy data file already exists: {dummy_file_path}")
        # To test with only dummy data, update pairs and timeframes:
        # pairs_to_validate = [dummy_pair_for_test]
        # timeframes_to_validate = [dummy_timeframe_for_test]


    for pair in pairs_to_validate:
        for timeframe in timeframes_to_validate:
            print(f"\n======================================================================")
            print(f"Processing: Pair: {pair}, Timeframe: {timeframe}, Exchange: {exchange_to_validate}")
            print(f"======================================================================")

            df_pair_data = load_data_for_pair(
                base_data_dir=base_data_dir_to_validate,
                exchange=exchange_to_validate,
                pair=pair,
                timeframe=timeframe
            )

            if df_pair_data is not None and not df_pair_data.empty:
                if validate_ohlcv_data(df_pair_data, pair, timeframe):
                    if ta: # Only calculate if ta is available
                        calculate_indicators(df_pair_data, pair, timeframe)
                    else:
                        print("\n--- Skipping Technical Indicators (ta library not found) ---")
                    check_cnn_data_suitability(df_pair_data, pair, timeframe)
            else:
                print(f"Skipping validation and checks for {pair} - {timeframe} due to loading error or empty data.")
            print("----------------------------------------------------------------------\n")

    print("Data validation script finished.")

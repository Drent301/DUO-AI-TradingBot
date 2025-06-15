import pandas as pd
from pathlib import Path # Ensure Path is imported directly
import json
import logging
import os # Keep os for os.getenv if used elsewhere, not directly here for paths.
import sys

# Configure logging using pathlib
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent # Assumes utils is directly under project root

LOG_DIR = PROJECT_ROOT_DIR / "user_data" / "logs"
LOG_FILE_PATH = LOG_DIR / "data_validator.log"

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, # Consider making this configurable
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data_for_pair(base_data_dir: str, exchange: str, pair: str, timeframe: str) -> pd.DataFrame | None:
    """
    Loads Freqtrade data for a specific pair and timeframe into a Pandas DataFrame.
    Freqtrade stores data in JSON files like <BASE_DATA_DIR>/<EXCHANGE>/<PAIR_WITH_UNDERSCORE>-<TIMEFRAME>.json
    e.g., user_data/data/binance/ETH_BTC-5m.json
    """
    base_data_dir_path = Path(base_data_dir)
    # If base_data_dir is relative, resolve it against project root for robustness,
    # common if script is called from project root and 'user_data/data' is passed.
    if not base_data_dir_path.is_absolute() and base_data_dir_path.parts[0] == 'user_data':
        base_data_dir_path = (PROJECT_ROOT_DIR / base_data_dir_path).resolve()
    elif not base_data_dir_path.is_absolute(): # Generic relative path
        base_data_dir_path = base_data_dir_path.resolve()


    pair_filename = pair.replace('/', '_')
    file_path = base_data_dir_path / exchange.lower() / f"{pair_filename}-{timeframe}.json"

    logger.info(f"\nAttempting to load data from: {file_path}")
    try:
        with file_path.open('r', encoding='utf-8') as f: # Use Path.open()
            data = json.load(f)

        if not data:
            logger.warning(f"File loaded but contains no data: {file_path}")
            return None

        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])

        # Convert 'date' from milliseconds to datetime
        df['date'] = pd.to_datetime(df['date'], unit='ms')

        # Standardize 'date' column to UTC
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize('UTC')
            logger.debug(f"Localized 'date' column to UTC for {file_path}")
        else:
            df['date'] = df['date'].dt.tz_convert('UTC')
            logger.debug(f"Converted 'date' column to UTC for {file_path}")

        df.set_index('date', inplace=True)

        # Ensure OHLCV column names are lowercase (already done by columns list above)
        # and convert to numeric. This also handles if source JSON had string numbers.
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

        logger.info(f"Successfully loaded and processed data for {pair} - {timeframe}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Data file not found: {file_path}", exc_info=True)
        return None
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from file: {file_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
        return None

def validate_ohlcv_data(df: pd.DataFrame, pair: str, timeframe: str) -> bool:
    """
    Validates basic OHLCV data integrity.
    """
    logger.info(f"\n--- Validating OHLCV data for {pair} - {timeframe} ---")
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Missing required columns: {missing_cols}", exc_info=False) # No direct exception
        return False

    if df.empty:
        logger.error(f"Validation FAILED for {pair} - {timeframe}: DataFrame is empty.", exc_info=False)
        return False

    if df[required_columns].isnull().any().any():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Contains NaN values in OHLCV columns.", exc_info=False)
        logger.error(df[required_columns].isnull().sum()) # Log sum of NaNs per column
        return False

    # Check if high is always >= low, open, close and low is always <= open, close
    if not (df['high'] >= df['low']).all():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Not all high values are greater than or equal to low values.", exc_info=False)
        return False
    if not (df['high'] >= df['open']).all():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Not all high values are greater than or equal to open values.", exc_info=False)
        return False
    if not (df['high'] >= df['close']).all():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Not all high values are greater than or equal to close values.", exc_info=False)
        return False
    if not (df['low'] <= df['open']).all():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Not all low values are less than or equal to open values.", exc_info=False)
        return False
    if not (df['low'] <= df['close']).all():
        logger.error(f"Validation FAILED for {pair} - {timeframe}: Not all low values are less than or equal to close values.", exc_info=False)
        return False

    logger.info("OHLCV data validation SUCCEEDED.")
    return True

def check_cnn_data_suitability(df: pd.DataFrame, pair: str, timeframe: str):
    """
    Provides descriptive information about CNN data suitability.
    """
    logger.info(f"\n--- CNN Data Suitability Check for {pair} - {timeframe} ---")
    if df.empty:
        logger.warning("DataFrame is empty. Cannot assess CNN suitability.")
        return

    logger.info(f"DataFrame Shape: {df.shape}")
    logger.info(f"DataFrame Columns: {df.columns.tolist()}")

    logger.info("\nConsiderations for CNN input (e.g., for core/cnn_patterns.py):")
    logger.info("- Ensure sufficient historical data points (sequence length) for each sample.")
    logger.info(f"  Current dataset length: {len(df)}")
    logger.info("- Data normalization or scaling (e.g., MinMaxScaler, StandardScaler) is typically required.")
    logger.info("- OHLCV data might need to be transformed into image-like structures (e.g., Gramian Angular Fields, recurrence plots) or specific feature sets.")
    logger.info("- The exact input requirements depend heavily on the CNN architecture defined in `core/cnn_patterns.py` (if applicable) or other model.")
    logger.info("- Features might include raw OHLCV, returns, or technical indicators.")

if __name__ == "__main__":
    pairs_to_validate = ["ZEN/BTC", "LSK/BTC", "ETH/BTC", "ETH/EUR"]
    timeframes_to_validate = ["1m", "5m", "15m", "1h", "4h", "1d"]
    # timeframes_to_validate = ["1h"] # For quicker testing
    exchange_to_validate = "binance"
    # This should be the directory containing the <exchange_name> subdirectories
    # e.g., user_data/data if files are in user_data/data/binance/
    # Define relative to project root for clarity
    base_data_dir_to_validate_path = PROJECT_ROOT_DIR / "user_data" / "data"

    logger.info(f"Starting data validation process for exchange: {exchange_to_validate}")
    logger.info(f"Using base data directory: {base_data_dir_to_validate_path}")
    logger.info("---")

    # Create dummy data for testing if no real data is present
    create_dummy_data = False # Set to True to create dummy files for a quick test run
    if create_dummy_data:
        logger.info("Attempting to create dummy data for testing...")
        dummy_pair_for_test = "ETH/BTC" # Example pair
        dummy_timeframe_for_test = "1h" # Example timeframe
        # Construct dummy_file_path using base_data_dir_to_validate_path (which is a Path object)
        dummy_file_path = base_data_dir_to_validate_path / exchange_to_validate.lower() / f"{dummy_pair_for_test.replace('/', '_')}-{dummy_timeframe_for_test}.json"
        dummy_file_path.parent.mkdir(parents=True, exist_ok=True) # Correct usage
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
            with dummy_file_path.open('w', encoding='utf-8') as f: # Use Path.open()
                json.dump(sample_json_data, f)
            logger.info(f"Created dummy data file: {dummy_file_path}")
        else:
            logger.info(f"Dummy data file already exists: {dummy_file_path}")
        # To test with only dummy data, update pairs and timeframes:
        # pairs_to_validate = [dummy_pair_for_test]
        # timeframes_to_validate = [dummy_timeframe_for_test]


    for pair in pairs_to_validate:
        for timeframe in timeframes_to_validate:
            logger.info(f"\n======================================================================")
            logger.info(f"Processing: Pair: {pair}, Timeframe: {timeframe}, Exchange: {exchange_to_validate}")
            logger.info(f"======================================================================")

            df_pair_data = load_data_for_pair(
                base_data_dir=str(base_data_dir_to_validate_path), # Pass as string, function converts to Path
                exchange=exchange_to_validate,
                pair=pair,
                timeframe=timeframe
            )

            if df_pair_data is not None and not df_pair_data.empty:
                if validate_ohlcv_data(df_pair_data, pair, timeframe):
                    # calculate_indicators was here
                    check_cnn_data_suitability(df_pair_data, pair, timeframe)
            else:
                logger.warning(f"Skipping validation and checks for {pair} - {timeframe} due to loading error or empty data.")
            logger.info("----------------------------------------------------------------------\n")

    logger.info("Data validation script finished.")

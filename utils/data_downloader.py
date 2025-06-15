# utils/data_downloader.py
"""
This script is used to download historical market data using Freqtrade.

When executed directly (e.g., `python utils/data_downloader.py` from the project root),
it will:
1. Load configuration from `config/config.json` (exchange name and pair whitelist).
2. Download 5 years of historical data (configurable by DAYS_TO_DOWNLOAD in the script).
3. Fetch data for the following timeframes: 1m, 5m, 15m, 1h, 4h, 1d (configurable by TARGET_TIMEFRAMES).
4. Store the data in the `user_data/data/<exchange_name>` directory.

Prerequisites:
- Freqtrade must be installed and accessible in your system's PATH.
  You can typically install it via pip: `pip install freqtrade`
- `config/config.json` must be configured with the target exchange (e.g., 'binance')
  and the desired 'pair_whitelist'. The script attempts to load this from the project root
  or a relative path if run from the `utils` directory.

To run:
Ensure you are in the project's root directory, then execute:
python utils/data_downloader.py
"""
import subprocess
import os # Keep for os.getenv if used, though not in this file.
from pathlib import Path # Import Path
import logging
import sys

# Configure logging using pathlib
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent # Assumes utils is directly under project root

LOG_DIR = PROJECT_ROOT_DIR / "user_data" / "logs"
LOG_FILE_PATH = LOG_DIR / "data_downloader.log"

# Create log directory if it doesn't exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, # Consider making this configurable or using logger.setLevel
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def download_data(pairs: list[str], timeframes: list[str], exchange: str, data_dir: str = "user_data/data/binance", days: int = None, since: str = None):
    """
    Downloads data using Freqtrade.
    Assumes data_dir is structured like 'path/to/user_data/data/exchange_name'.
    The --datadir for Freqtrade will then be 'path/to/user_data'.
    """
    data_dir_path = Path(data_dir)

    # Determine Freqtrade's --datadir.
    # If data_dir is "user_data/data/binance", freqtrade_datadir should be "user_data".
    try:
        if not data_dir_path.is_absolute():
            if data_dir_path.parts and data_dir_path.parts[0] == 'user_data':
                 data_dir_path = (PROJECT_ROOT_DIR / data_dir_path).resolve()
            else:
                 data_dir_path = data_dir_path.resolve()

        # Simplified logic: Assume data_dir is <path_to_user_data>/data/<exchange_name>
        # So, freqtrade_datadir is <path_to_user_data>
        if data_dir_path.parent.name == 'data':
            freqtrade_datadir = data_dir_path.parent.parent
        else:
            logger.warning(
                f"Unexpected data_dir structure: '{data_dir_path}'. "
                f"Expected <path_to_user_data>/data/<exchange_name>. "
                f"Falling back to using '{data_dir_path.parent}' as Freqtrade --datadir. "
                f"This might be incorrect if '{data_dir_path.parent}/data' does not exist."
            )
            freqtrade_datadir = data_dir_path.parent

        # Ensure the determined freqtrade_datadir actually contains a 'data' subdirectory,
        # as Freqtrade expects this structure for many operations.
        if not (freqtrade_datadir / "data").is_dir():
            logger.warning(
                f"The determined Freqtrade --datadir '{freqtrade_datadir}' does not appear to contain a 'data' subdirectory. "
                f"Freqtrade might not work as expected. Please verify your data_dir structure and Freqtrade's requirements."
            )

    except Exception as e:
        logger.error(f"Error determining Freqtrade --datadir from data_dir '{data_dir}': {e}", exc_info=True)
        return

    logger.info(f"Using Freqtrade --datadir: '{str(freqtrade_datadir)}' for specified data_dir: '{data_dir_path}'")

    # Create the specific data directory if it doesn't exist
    try:
        data_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured data directory exists: {data_dir_path}")
    except OSError as e:
        logger.error(f"Error creating directory {data_dir_path}: {e}", exc_info=True)
        return

    for pair in pairs:
        for timeframe in timeframes:
            command = [
                "freqtrade", "download-data",
                "--exchange", exchange.lower(),
                "-p", pair,
                "-t", timeframe,
                "--datadir", str(freqtrade_datadir) # subprocess needs string paths
            ]

            if days is not None:
                command.extend(["--days", str(days)])
            elif since is not None:
                command.extend(["--since", since])
            else:
                logger.warning(
                    f"Neither 'days' nor 'since' provided for downloading {pair} - {timeframe}. "
                    f"Freqtrade's default download period will apply."
                )
                # Freqtrade download-data defaults to 30 days if neither --days nor --since is given.

            logger.info(f"Executing command: {' '.join(command)}")
            try:
                process = subprocess.run(command, check=True, capture_output=True, text=True)
                logger.info(f"Successfully downloaded data for {pair} - {timeframe}")
                if process.stdout:
                    logger.info("Stdout:")
                    logger.info(process.stdout)
                if process.stderr:
                    logger.warning("Stderr:")
                    logger.warning(process.stderr)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error downloading data for {pair} - {timeframe} on {exchange}.")
                logger.error(f"Command failed: {' '.join(e.cmd)}")
                logger.error(f"Return code: {e.returncode}")
                if e.stdout:
                    logger.error("Stdout:")
                    logger.error(e.stdout)
                if e.stderr:
                    logger.error("Stderr:")
                    logger.error(e.stderr)
            except FileNotFoundError:
                logger.error("Error: 'freqtrade' command not found. Make sure Freqtrade is installed and in your PATH.")
                return # Stop further processing if freqtrade is not found

if __name__ == "__main__":
    import json

    TARGET_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    DAYS_TO_DOWNLOAD = 5 * 365
    CONFIG_PATH_PROJECT_ROOT = "config/config.json"
    CONFIG_PATH_RELATIVE_TO_UTILS = "../config/config.json" # If script is in utils/

    config_data = None
    try:
        with open(CONFIG_PATH_PROJECT_ROOT, 'r') as f:
            config_data = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH_PROJECT_ROOT}")
    except FileNotFoundError:
        try:
            with open(CONFIG_PATH_RELATIVE_TO_UTILS, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from {CONFIG_PATH_RELATIVE_TO_UTILS}")
        except FileNotFoundError:
            logger.error(f"Error: Configuration file not found at {CONFIG_PATH_PROJECT_ROOT} or {CONFIG_PATH_RELATIVE_TO_UTILS}")
            exit(1) # Exit if config not found
    except json.JSONDecodeError:
        logger.error("Error: Could not decode JSON from the configuration file.")
        exit(1)


    exchange_name = config_data['exchange']['name']
    pairs_to_download = config_data['exchange']['pair_whitelist']
    data_directory = f"user_data/data/{exchange_name.lower()}"

    logger.info(f"--- Data Download Configuration ---")
    logger.info(f"Exchange: {exchange_name}")
    logger.info(f"Pairs: {pairs_to_download}")
    logger.info(f"Timeframes: {TARGET_TIMEFRAMES}")
    logger.info(f"Data Directory: {data_directory}")
    logger.info(f"Days to Download: {DAYS_TO_DOWNLOAD}")
    logger.info(f"-----------------------------------")

    download_data(
        pairs=pairs_to_download,
        timeframes=TARGET_TIMEFRAMES,
        exchange=exchange_name,
        data_dir=data_directory,
        days=DAYS_TO_DOWNLOAD
    )
    logger.info("Data download process finished.")

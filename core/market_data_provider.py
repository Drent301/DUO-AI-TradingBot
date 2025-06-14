# core/market_data_provider.py
import logging
import pandas as pd
from typing import Optional
import os # For environment variables if needed for paths
import asyncio # For asyncio.to_thread
from datetime import datetime, timedelta # For date calculations

# Attempt to import from utils, assuming they are in PYTHONPATH
try:
    from utils.data_validator import load_data_for_pair
    from utils.data_downloader import download_data as freqtrade_download_data
except ImportError as e:
    logging.error(f"MarketDataProvider: Failed to import utils: {e}. Market data functionality will be limited.")
    # Define dummy functions if imports fail, so AIOptimizer can at least import this module
    def load_data_for_pair(*args, **kwargs) -> Optional[pd.DataFrame]:
        logging.error("MarketDataProvider: load_data_for_pair (dummy) called due to import error.")
        return None
    def freqtrade_download_data(*args, **kwargs):
        logging.error("MarketDataProvider: freqtrade_download_data (dummy) called due to import error.")
        pass

logger = logging.getLogger(__name__)

DEFAULT_BASE_DATA_DIR = "user_data/data" # Default location for Freqtrade data

async def get_recent_market_data(
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    days_to_load: int = 90, # How many days of data the AI optimizer might need
    download_if_missing: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetches recent market data for a given symbol and timeframe.
    Tries to load from local Freqtrade data storage first.
    If data is missing or insufficient, and download_if_missing is True,
    it attempts to download the required data using Freqtrade's download script.

    Args:
        symbol (str): The trading symbol (e.g., "ETH/USDT").
        timeframe (str): The timeframe (e.g., "5m", "1h").
        exchange (str): The exchange name (default: "binance").
        days_to_load (int): The number of recent days of data to try and ensure are available.
        download_if_missing (bool): Whether to attempt downloading if data is not found.

    Returns:
        Optional[pd.DataFrame]: DataFrame with OHLCV data, or None if data cannot be obtained.
    """
    logger.info(f"Attempting to get recent market data for {exchange} - {symbol} - {timeframe} ({days_to_load} days).")

    base_data_dir = os.getenv("FREQTRADE_USER_DATA_DIR", "user_data") + "/data"
    exchange_specific_data_dir = f"{base_data_dir}/{exchange.lower()}"

    df = await asyncio.to_thread(
        load_data_for_pair,
        base_data_dir=base_data_dir,
        exchange=exchange.lower(),
        pair=symbol,
        timeframe=timeframe
    )

    if df is not None and not df.empty:
        required_start_date = datetime.utcnow() - timedelta(days=days_to_load)
        # Ensure DataFrame index is timezone-aware (UTC) or naive, consistent with required_start_date
        # If df.index is naive, make required_start_date naive for comparison.
        # If df.index is aware, ensure required_start_date is also aware (utcnow() is timezone-aware).
        # For simplicity, assuming df.index from Freqtrade is UTC.

        if df.index.max() > required_start_date: # Check if latest data is recent enough
            # Check if data goes back far enough
            if df.index.min() <= required_start_date:
                logger.info(f"Sufficient existing data found for {symbol} - {timeframe} covering the period.")
                cutoff_date_for_return = datetime.utcnow() - timedelta(days=days_to_load)
                df_recent = df[df.index >= cutoff_date_for_return]
                return df_recent if not df_recent.empty else None # Should not be empty if previous checks passed
            else:
                logger.info(f"Existing data for {symbol} - {timeframe} is recent but does not go back {days_to_load} days (oldest is {df.index.min()}). Will attempt download if allowed.")
                # Don't set df to None here if download_if_missing is False, partial data might be acceptable
                if not download_if_missing:
                    cutoff_date_for_return = datetime.utcnow() - timedelta(days=days_to_load)
                    df_recent = df[df.index >= cutoff_date_for_return] # Return what we have that's recent
                    return df_recent if not df_recent.empty else None
                # If download is allowed, proceed to download logic by falling through or explicitly setting df = None
                df = None # Force download to get older data

        else: # All existing data is older than required_start_date
            logger.info(f"Existing data for {symbol} - {timeframe} is too old (latest is {df.index.max()}). Will attempt download if allowed.")
            if not download_if_missing:
                return None # Too old and no download allowed
            df = None # Force download attempt

    if df is None and download_if_missing:
        logger.info(f"Data for {symbol} - {timeframe} not found or insufficient. Attempting download for the last {days_to_load + 5} days.")
        try:
            await asyncio.to_thread(
                freqtrade_download_data,
                pairs=[symbol],
                timeframes=[timeframe],
                exchange=exchange.lower(),
                data_dir=exchange_specific_data_dir,
                days=days_to_load + 5 # Add a small buffer for download
            )
            logger.info(f"Download attempt finished for {symbol} - {timeframe}. Reloading data.")
            df = await asyncio.to_thread(
                load_data_for_pair,
                base_data_dir=base_data_dir,
                exchange=exchange.lower(),
                pair=symbol,
                timeframe=timeframe
            )
        except Exception as e:
            logger.error(f"Error during data download or reload for {symbol} - {timeframe}: {e}", exc_info=True)
            return None

    if df is not None and not df.empty:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_load)
        df_recent = df[df.index >= cutoff_date]
        if not df_recent.empty:
            logger.info(f"Successfully obtained recent market data for {symbol} - {timeframe}. Shape: {df_recent.shape}")
            return df_recent
        else:
            logger.warning(f"Data obtained for {symbol} - {timeframe}, but it's all older than {days_to_load} days.")
            return None
    else:
        logger.warning(f"Could not obtain market data for {symbol} - {timeframe} after load/download attempts.")
        return None

async def example_usage():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # This example assumes that if utils cannot be imported, the dummy functions will log errors
    # but allow the script to run. For actual data fetching, utils must be available.

    symbol_to_test = "ETH/USDT"
    timeframe_to_test = "1h"
    exchange_to_test = "binance"

    logger.info(f"--- Example: Fetching data for {symbol_to_test} {timeframe_to_test} (download allowed) ---")
    data = await get_recent_market_data(
        symbol_to_test,
        timeframe_to_test,
        exchange_to_test,
        days_to_load=30,
        download_if_missing=True
    )

    if data is not None:
        logger.info(f"Received data for {symbol_to_test} {timeframe_to_test}: Shape: {data.shape}, Start: {data.index.min()}, End: {data.index.max()}")
        # logger.info(f"Head:\n{data.head()}") # Can be verbose
    else:
        logger.info(f"No data received for {symbol_to_test} {timeframe_to_test}.")

    logger.info(f"--- Example: Fetching data for a potentially non-existent pair (NO download) ---")
    symbol_non_existent = "NONEXISTENT/PAIRTHATSHOULDNOTEXIST"
    data_no_download = await get_recent_market_data(
        symbol_non_existent,
        "1h",
        exchange_to_test,
        days_to_load=7,
        download_if_missing=False
    )
    if data_no_download is not None:
        logger.error(f"Unexpectedly received data for {symbol_non_existent} when download was false.")
    else:
        logger.info(f"Correctly received no data for {symbol_non_existent} when download_if_missing=False.")

    logger.info(f"--- Example: Fetching data again for {symbol_to_test} {timeframe_to_test} (should be cached or load fast, no download if recent) ---")
    data_again = await get_recent_market_data(
        symbol_to_test,
        timeframe_to_test,
        exchange_to_test,
        days_to_load=30, # Request same 30 days
        download_if_missing=True # Download allowed, but hopefully not needed if first call worked
    )
    if data_again is not None:
         logger.info(f"Received data again for {symbol_to_test} {timeframe_to_test}: Shape: {data_again.shape}, Start: {data_again.index.min()}, End: {data_again.index.max()}")
    else:
        logger.info(f"No data received for {symbol_to_test} {timeframe_to_test} on second attempt.")


if __name__ == "__main__":
    # This ensures that if this script is run directly, the example usage is executed.
    # Note: For the download part to work, Freqtrade must be installed and configured,
    # and its dependencies (like TA-Lib) must be available if data_downloader uses them indirectly.
    # The `utils` imports also need to be resolvable from the current execution path.
    # (e.g., by running from the project root or having PYTHONPATH set up)
    asyncio.run(example_usage())

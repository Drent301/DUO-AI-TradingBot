import logging
import asyncio
import os
import sys
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv() # Load default .env file

# Util imports
from utils.data_downloader import download_data
from utils.data_validator import load_data_for_pair, validate_ohlcv_data, calculate_indicators, check_cnn_data_suitability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_run.log"), # Log to a file
        logging.StreamHandler(sys.stdout) # Log also to console
    ]
)
logger = logging.getLogger(__name__)

# Core component imports
try:
    from core.params_manager import ParamsManager
    from core.bitvavo_executor import BitvavoExecutor
    from core.cnn_patterns import CNNPatterns
    from core.pre_trainer import PreTrainer
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)


async def main():
    logger.info("======================================================================")
    logger.info("=== Starting Full Data Fetching, Pre-Training, & Backtesting Pipeline ===")
    logger.info("======================================================================")

    parser = argparse.ArgumentParser(description="Run the AI trading pipeline.")
    parser.add_argument('--download-data', action='store_true', help='Download historical market data using Freqtrade.')
    parser.add_argument('--validate-data', action='store_true', help='Validate downloaded market data.')
    # Potentially add other arguments from core components if needed, or let them use ParamsManager
    args = parser.parse_args()

    if args.download_data:
        logger.info("--- Initiating Data Download Process ---")
        pairs_to_download = ["ZEN/BTC", "LSK/BTC", "ETH/BTC", "ETH/EUR"] # As defined in previous steps
        timeframes_to_download = ["1m", "5m", "15m", "1h", "4h", "1d"]
        exchange_to_download = "binance"
        days_to_download = 5 * 365
        # data_dir for download_data is the specific exchange data directory, e.g., user_data/data/binance
        # download_data function defaults to "user_data/data/binance"
        # Explicitly setting for clarity and consistency with problem description
        downloader_data_dir = f"user_data/data/{exchange_to_download.lower()}"

        try:
            download_data(
                pairs=pairs_to_download,
                timeframes=timeframes_to_download,
                exchange=exchange_to_download,
                data_dir=downloader_data_dir,
                days=days_to_download
            )
            logger.info("--- Data Download Process Finished ---")
        except Exception as e:
            logger.error(f"Error during data download process: {e}", exc_info=True)
            # Decide if pipeline should stop if download fails. For now, it will continue.

    if args.validate_data:
        logger.info("--- Initiating Data Validation Process ---")
        pairs_to_validate = ["ZEN/BTC", "LSK/BTC", "ETH/BTC", "ETH/EUR"] # As defined in previous steps
        timeframes_to_validate = ["1m", "5m", "15m", "1h", "4h", "1d"]
        exchange_to_validate = "binance"
        # base_data_dir for load_data_for_pair is the root of exchange data, e.g., user_data/data
        base_data_dir_for_validation = "user_data/data"

        for pair in pairs_to_validate:
            for timeframe in timeframes_to_validate:
                logger.info(f"--- Validating data for {pair} - {timeframe} on {exchange_to_validate} ---")
                df_pair_data = load_data_for_pair(
                    base_data_dir=base_data_dir_for_validation,
                    exchange=exchange_to_validate,
                    pair=pair,
                    timeframe=timeframe
                )
                if df_pair_data is not None and not df_pair_data.empty:
                    # Pass pair and timeframe to validation functions as they expect them
                    if validate_ohlcv_data(df_pair_data, pair, timeframe):
                        calculate_indicators(df_pair_data, pair, timeframe)
                        check_cnn_data_suitability(df_pair_data, pair, timeframe)
                else:
                    logger.warning(f"No data loaded for {pair} - {timeframe}. Skipping further validation for this item.")
        logger.info("--- Data Validation Process Finished ---")

    # Initialize Core Components
    logger.info("Initializing core components...")
    try:
        params_manager = ParamsManager()

        # Attempt to initialize BitvavoExecutor, handle missing API keys
        try:
            bitvavo_executor = BitvavoExecutor()
            logger.info("BitvavoExecutor initialized successfully.")
        except ValueError as e:
            logger.error(f"Failed to initialize BitvavoExecutor: {e}")
            logger.error("Pipeline execution will be limited (e.g., no live data fetching or backtesting with live data).")
            # Depending on requirements, you might want to exit:
            # sys.exit(1)
            # For now, we'll allow it to continue for components that don't strictly need live keys (e.g. loading cached data)
            bitvavo_executor = None

        # CNNPatterns now requires params_manager in its constructor
        cnn_pattern_detector = CNNPatterns(params_manager=params_manager)
        logger.info("CNNPatterns detector initialized successfully.")

        # PreTrainer requires params_manager, cnn_pattern_detector, and bitvavo_executor
        pre_trainer = PreTrainer(
            params_manager=params_manager,
            cnn_pattern_detector=cnn_pattern_detector,
            bitvavo_executor=bitvavo_executor
        )
        logger.info("PreTrainer initialized successfully.")

    except Exception as e:
        logger.error(f"Fatal error during core component initialization: {e}", exc_info=True)
        sys.exit(1)

    # Run the Pre-training Pipeline
    try:
        logger.info("--- Running Pre-training & Backtesting Pipeline ---")

        # Get the strategy ID to run from ParamsManager or use a default
        # (This default_strategy_id should ideally be part of your ParamsManager defaults)
        strategy_id_to_run = params_manager.get_param('default_strategy_id')
        if not strategy_id_to_run:
            strategy_id_to_run = "DefaultPipelineRunStrategy" # Fallback if not in params
            logger.warning(f"'default_strategy_id' not found in ParamsManager, using fallback: {strategy_id_to_run}")

        await pre_trainer.run_pretraining_pipeline(strategy_id=strategy_id_to_run)
        logger.info("--- Pre-training & Backtesting Pipeline completed. ---")

    except Exception as e:
        logger.error(f"Error during pre-training pipeline execution: {e}", exc_info=True)
        # Depending on the error, you might want to exit or attempt cleanup

    logger.info("===================================================================")
    logger.info("=== Full Data and Training Pipeline Run Finished Successfully ===")
    logger.info("===================================================================")

if __name__ == "__main__":
    # Check for environment variables needed by BitvavoExecutor early
    # This is just a preliminary check; BitvavoExecutor itself will raise ValueError
    api_key = os.getenv('BITVAVO_API_KEY')
    secret_key = os.getenv('BITVAVO_SECRET_KEY')

    if not api_key or not secret_key:
        logger.warning("BITVAVO_API_KEY or BITVAVO_SECRET_KEY not found in .env file.")
        logger.warning("BitvavoExecutor may not initialize correctly, limiting live data operations.")
        # Allow to proceed, as some parts of the pipeline might work with cached data or without Bitvavo.

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled critical error at top level: {e}", exc_info=True)
        sys.exit(1)

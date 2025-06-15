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
# VERWIJDERD: calculate_indicators uit deze importregel
from utils.data_validator import load_data_for_pair, validate_ohlcv_data, check_cnn_data_suitability
import pandas as pd

# Configure logging
# AANGEPAST: Schrijf logbestanden naar user_data/logs/pipeline_run.log
log_dir = "user_data/logs"
log_file_path = os.path.join(log_dir, "pipeline_run.log")

# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to a file
        logging.StreamHandler(sys.stdout) # Log also to console
    ]
)
logger = logging.getLogger(__name__)

# Attempt to import Freqtrade-dependent modules, handle if Freqtrade is not installed
try:
    from strategies.DUOAI_Strategy import DUOAI_Strategy
    from freqtrade.data.dataprovider import DataProvider
    from freqtrade.configuration import Configuration
    FREQTRADE_AVAILABLE = True
except ImportError:
    logger.warning("Freqtrade modules not found. Some functionalities (like main strategy execution or Freqtrade-dependent data processing in pretrain) will be unavailable.")
    DUOAI_Strategy = None
    DataProvider = None
    Configuration = None
    FREQTRADE_AVAILABLE = False

# Core component imports
try:
    from core.params_manager import ParamsManager
    from core.bitvavo_executor import BitvavoExecutor
    from core.cnn_patterns import CNNPatterns
    from core.pre_trainer import PreTrainer
    from core.reflectie_lus import ReflectieLus
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
    parser.add_argument('--pretrain-cnn', action='store_true', help='Run CNN pre-training process.')
    parser.add_argument('--pretrain-pair', type=str, default='ETH/BTC', help='Pair to use for CNN pre-training (e.g., ETH/BTC).')
    parser.add_argument('--pretrain-timeframe', type=str, default='5m', help='Timeframe to use for CNN pre-training (e.g., 5m).')
    parser.add_argument('--start-reflection-loop', action='store_true', help='Start the continuous AI reflection loop.')
    parser.add_argument('--reflection-symbols', type=str, default='ETH/USDT,BTC/USDT', help='Comma-separated list of symbols for the reflection loop (e.g., ETH/USDT,BTC/USDT).')
    parser.add_argument('--reflection-interval', type=int, default=60, help='Interval in minutes for the reflection loop.')
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
                        # VERWIJDERD: De aanroep naar calculate_indicators
                        check_cnn_data_suitability(df_pair_data, pair, timeframe)
                else:
                    logger.warning(f"No data loaded for {pair} - {timeframe}. Skipping further validation for this item.")
        logger.info("--- Data Validation Process Finished ---")

    if args.pretrain_cnn:
        logger.info(f"--- Starting CNN Pre-training Pipeline ---")
        try:
            logger.info("Instantiating ParamsManager for pre-training...")
            # ParamsManager loads its settings from JSON files within its constructor
            params_manager = ParamsManager(user_data_dir="user_data")
            # Potentially update params_manager with CLI args if needed, e.g.:
            # params_manager.update_specific_params({'pretrain_pair': args.pretrain_pair, 'pretrain_timeframe': args.pretrain_timeframe})
            # However, the idea is that ParamsManager itself defines these for the pipeline.

            logger.info("Instantiating PreTrainer...")
            # PreTrainer needs a configuration. This might include exchange details if BitvavoExecutor is used.
            # Assuming PreTrainer's config can take 'user_data_dir' and potentially other relevant settings.
            # If BitvavoExecutor is initialized by PreTrainer, it might need 'exchange_name' or similar.
            # For now, providing a basic config. PreTrainer should be designed to fetch further
            # specific params it needs from the params_manager instance.
            pre_trainer_config = {
                'user_data_dir': "user_data",
                'exchange_name': "binance", # Example, if PreTrainer initializes BitvavoExecutor
                # Add other necessary basic configs for PreTrainer initialization if any.
            }
            pre_trainer = PreTrainer(config=pre_trainer_config)

            strategy_id = "DuoAI_Strategy_Pretrain" # Default strategy ID
            logger.info(f"Running pre-training pipeline for strategy_id: {strategy_id}...")

            # run_pretraining_pipeline is an async method, so it needs to be run in an event loop.
            # Since main() is already async and run with asyncio.run() at the script's end,
            # we can await it here directly.
            await pre_trainer.run_pretraining_pipeline(
                strategy_id=strategy_id,
                params_manager=params_manager
            )
            logger.info(f"--- CNN Pre-training Pipeline for strategy_id: {strategy_id} Finished ---")

        except Exception as e:
            logger.error(f"Error during CNN pre-training pipeline: {e}", exc_info=True)
            logger.info(f"--- CNN Pre-training Pipeline Failed ---")

    # Initialize Core Components
    # Note: If --pretrain-cnn is the only task, other initializations might not be needed or
    # should be conditional. The current structure seems to allow this.
    logger.info("Initializing core components...")
    try:
        # ParamsManager might be initialized here if needed globally,
        # but for pre-training, it's handled within the --pretrain-cnn block.
        # DUOAI_Strategy initializes its own ParamsManager.
        if not args.pretrain_cnn: # Only run main pipeline components if not in pre-training mode
            logger.info("Skipping main trading pipeline initialization as --pretrain-cnn was specified.")
            # If there were other components to initialize here for a non-pretrain run, they would go here.
        else:
            # If --pretrain-cnn was run, this section for other core components might be skipped or adjusted
            logger.info("Pre-training was run. Main pipeline component initialization might be skipped or adjusted.")
            pass


    except Exception as e:
        logger.error(f"Fatal error during (potential) core component initialization for main pipeline: {e}", exc_info=True)
        # Decide if sys.exit(1) is appropriate here. If pre-training succeeded, maybe not.
        # For now, let it continue to the end of the script.

    # Main pipeline execution logic (e.g., running DUOAI_Strategy)
    # This should only run if not in a specific pre-training mode.
    if not args.pretrain_cnn:
        logger.info("--- Attempting to run main trading pipeline (DUOAI_Strategy flow, if configured) ---")
        # Example:
        # if FREQTRADE_AVAILABLE and DUOAI_Strategy:
        #     logger.info("Freqtrade is available. To run the main strategy, you would typically use Freqtrade's CLI.")
        #     logger.info("e.g., freqtrade trade --config user_data/configs/config_duoai.json --strategy DUOAI_Strategy")
        # else:
        #     logger.info("Freqtrade or DUOAI_Strategy not available. Main trading pipeline execution is skipped.")
        logger.info("Main pipeline execution (strategy's own PreTrainer, etc.) is typically handled by Freqtrade CLI or a dedicated strategy runner.")
        logger.info("This script focuses on utility tasks like data download, validation, and the new pre-training pipeline.")

    logger.info("===================================================================")
    logger.info("=== Pipeline Run Finished ===") # Adjusted message
    logger.info("===================================================================")

    if args.start_reflection_loop:
        logger.info("--- Initiating AI Reflection Loop ---")
        logger.info("The script will now enter the reflection loop. Other operations will not proceed past this point if this is the final operation.")
        try:
            parsed_symbols = [symbol.strip() for symbol in args.reflection_symbols.split(',')]
            if not parsed_symbols or all(s == '' for s in parsed_symbols):
                logger.error("Reflection symbols are empty or invalid. Please provide a comma-separated list, e.g., --reflection-symbols ETH/USDT,BTC/USDT")
            else:
                reflection_loop = ReflectieLus() # Assuming ReflectieLus() doesn't require params_manager or other complex init for now
                # Ensure main() is async to use await here.
                await reflection_loop.start_reflection_loop(symbols=parsed_symbols, interval_minutes=args.reflection_interval)
                # The line above is an infinite loop, so code below this in the if block won't be reached.
        except Exception as e:
            logger.error(f"Error during reflection loop initialization or execution: {e}", exc_info=True)
        logger.info("--- AI Reflection Loop Finished (or encountered an error) ---") # This line might only be reached if the loop breaks due to error

if __name__ == "__main__":
    # Check for environment variables needed by BitvavoExecutor early
    # This is just a preliminary check; BitvavoExecutor itself will raise ValueError
    api_key = os.getenv('BITVAVO_API_KEY')
    secret_key = os.getenv('BITVAVO_SECRET_KEY')

    # Note: The check for api_key and secret_key is a general warning.
    # Specific components like BitvavoExecutor (if used by PreTrainer or ReflectieLus)
    # will handle their own configuration and raise errors if critical keys are missing.
    if not api_key or not secret_key:
        logger.warning("BITVAVO_API_KEY or BITVAVO_SECRET_KEY not found in .env file.")
        logger.warning("Components requiring Bitvavo API access may not initialize or function correctly.")

    try:
        asyncio.run(main()) # main() is already defined as an async function
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled critical error at top level: {e}", exc_info=True)
        sys.exit(1)

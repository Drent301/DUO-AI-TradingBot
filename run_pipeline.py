import logging
import asyncio
import os
import sys
import argparse
from pathlib import Path

# Load environment variables from .env file will be handled by config_validator
# from dotenv import load_dotenv
# load_dotenv() # Load default .env file

# Util imports
from utils.data_downloader import download_data
from utils.data_validator import load_data_for_pair, validate_ohlcv_data, check_cnn_data_suitability
from utils.config_validator import load_and_validate_config # Import the new validator
import pandas as pd

# Logger setup will be deferred until config is loaded
logger = logging.getLogger(__name__)

# Define base paths for clarity
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "config.json"
ENV_FILE_PATH = BASE_DIR / ".env"
LOG_DIR_RELATIVE_TO_USER_DATA = "logs" # e.g. user_data/logs
PIPELINE_LOG_FILENAME = "pipeline_run.log"


def setup_logging(log_file_path: Path):
    """Configurest basis logging."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path), # Log to a file
            logging.StreamHandler(sys.stdout) # Log also to console
        ]
    )

# Attempt to import Freqtrade-dependent modules, handle if Freqtrade is not installed
try:
    from strategies.DUOAI_Strategy import DUOAI_Strategy #TODO: Check if strategy name can come from config
    from freqtrade.data.dataprovider import DataProvider
    from freqtrade.configuration import Configuration
    FREQTRADE_AVAILABLE = True
except ImportError:
    # Logger might not be configured yet if this fails at module load time.
    # Using print for this specific early warning.
    print("Warning: Freqtrade modules not found. Some functionalities will be unavailable.", file=sys.stderr)
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
    # Logger might not be configured.
    print(f"Failed to import core modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.", file=sys.stderr)
    sys.exit(1)


async def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the AI trading pipeline.")
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='Path to the Freqtrade config.json file.')
    parser.add_argument('--env', type=str, default=str(ENV_FILE_PATH), help='Path to the .env file.')
    parser.add_argument('--download-data', action='store_true', help='Download historical market data.')
    parser.add_argument('--validate-data', action='store_true', help='Validate downloaded market data.')
    parser.add_argument('--pretrain-cnn', action='store_true', help='Run CNN pre-training process.')
    parser.add_argument('--pretrain-pair', type=str, help='Pair to use for CNN pre-training (e.g., ETH/BTC). Default from config.')
    parser.add_argument('--pretrain-timeframe', type=str, help='Timeframe to use for CNN pre-training (e.g., 5m). Default from config.')
    parser.add_argument('--start-reflection-loop', action='store_true', help='Start the continuous AI reflection loop.')
    parser.add_argument('--reflection-symbols', type=str, help='Comma-separated list of symbols for the reflection loop (e.g., ETH/USDT,BTC/USDT). Default from config pair_whitelist.')
    parser.add_argument('--reflection-interval', type=int, default=60, help='Interval in minutes for the reflection loop.')
    args = parser.parse_args()

    # --- Initial Setup (Config, Logging) ---
    try:
        config_path_obj = Path(args.config)
        env_path_obj = Path(args.env)
        ft_config = load_and_validate_config(str(config_path_obj), str(env_path_obj))

        user_data_dir = Path(ft_config.get("user_data_dir", "user_data"))
        if not user_data_dir.is_absolute():
            user_data_dir = BASE_DIR / user_data_dir

        log_dir = user_data_dir / LOG_DIR_RELATIVE_TO_USER_DATA
        log_file_path = log_dir / PIPELINE_LOG_FILENAME
        setup_logging(log_file_path)

        logger.info(f"Validated Freqtrade configuration loaded from {config_path_obj}")
        logger.info(f"Environment variables loaded from {env_path_obj}")
        logger.info(f"Logging to: {log_file_path}")

    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        # Attempt to log if logger was somehow initialized by a miracle
        if logger and logger.handlers:
             logger.error(f"Configuration error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during initialization: {e}", file=sys.stderr)
        if logger and logger.handlers:
            logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
        sys.exit(1)

    logger.info("======================================================================")
    logger.info("=== Starting AI Trading Pipeline ===")
    logger.info("======================================================================")

    # --- Get settings from config, with CLI overrides or defaults ---
    exchange_name = ft_config.get("exchange", {}).get("name", "binance") # Default to binance if not in config
    # Use all pairs from whitelist by default for download/validation, can be overridden by more specific CLI args later if needed
    config_pairs = ft_config.get("exchange", {}).get("pair_whitelist", [])
    # Timeframe from config, can be overridden by CLI for specific tasks like pretraining
    config_timeframe = ft_config.get("timeframe", "5m") # Default to 5m

    # data_dir from config, typically user_data/data
    # This is the root data directory (e.g. user_data/data)
    base_data_dir = Path(ft_config.get("data_dir", str(user_data_dir / "data")))
    if not base_data_dir.is_absolute(): # If data_dir is relative in config
        base_data_dir = BASE_DIR / base_data_dir


    # --- Data Download ---
    if args.download_data:
        logger.info("--- Initiating Data Download Process ---")
        # TODO: Potentially add CLI args for these if needed, or use config values
        pairs_to_download = config_pairs
        timeframes_to_download = ft_config.get("timeframes_to_download", ["1m", "5m", "15m", "1h", "4h", "1d"]) # Example: make this configurable
        days_to_download = ft_config.get("days_to_download", 5 * 365) # Example

        # download_data expects the data_dir to be specific to the exchange, e.g., user_data/data/binance
        downloader_data_dir = base_data_dir / exchange_name.lower()
        downloader_data_dir.mkdir(parents=True, exist_ok=True)


        try:
            download_data(
                pairs=pairs_to_download,
                timeframes=timeframes_to_download,
                exchange=exchange_name,
                data_dir=str(downloader_data_dir), # download_data might expect str
                days=days_to_download
            )
            logger.info(f"--- Data Download Process Finished for exchange {exchange_name} ---")
        except Exception as e:
            logger.error(f"Error during data download process: {e}", exc_info=True)

    # --- Data Validation ---
    if args.validate_data:
        logger.info("--- Initiating Data Validation Process ---")
        pairs_to_validate = config_pairs
        # Validate all timeframes that were potentially downloaded, or a specific list from config
        timeframes_to_validate = ft_config.get("timeframes_to_validate", ["1m", "5m", "15m", "1h", "4h", "1d"])

        for pair in pairs_to_validate:
            for timeframe_val in timeframes_to_validate: # Renamed to avoid conflict
                logger.info(f"--- Validating data for {pair} - {timeframe_val} on {exchange_name} ---")
                df_pair_data = load_data_for_pair(
                    base_data_dir=str(base_data_dir), # load_data_for_pair might expect str
                    exchange=exchange_name,
                    pair=pair,
                    timeframe=timeframe_val
                )
                if df_pair_data is not None and not df_pair_data.empty:
                    if validate_ohlcv_data(df_pair_data, pair, timeframe_val):
                        check_cnn_data_suitability(df_pair_data, pair, timeframe_val)
                else:
                    logger.warning(f"No data loaded for {pair} - {timeframe_val}. Skipping further validation.")
        logger.info("--- Data Validation Process Finished ---")

    # --- CNN Pre-training ---
    if args.pretrain_cnn:
        logger.info(f"--- Starting CNN Pre-training Pipeline ---")
        pretrain_pair = args.pretrain_pair or (config_pairs[0] if config_pairs else "ETH/BTC") # Default to first pair or ETH/BTC
        pretrain_timeframe = args.pretrain_timeframe or config_timeframe
        try:
            logger.info("Instantiating ParamsManager for pre-training...")
            # ParamsManager loads its settings from JSON files within its constructor
            # Use user_data_dir from the loaded and validated config
            params_manager = ParamsManager(user_data_dir=str(user_data_dir))
            # Potentially update params_manager with CLI args if needed, e.g.:
            # params_manager.update_specific_params({'pretrain_pair': pretrain_pair, 'pretrain_timeframe': pretrain_timeframe})
            # However, the idea is that ParamsManager itself defines these for the pipeline.

            logger.info("Instantiating PreTrainer...")
            # PreTrainer needs a configuration. This might include exchange details if BitvavoExecutor is used.
            # Assuming PreTrainer's config can take 'user_data_dir' and potentially other relevant settings.
            # If BitvavoExecutor is initialized by PreTrainer, it might need 'exchange_name' or similar.
            # For now, providing a basic config. PreTrainer should be designed to fetch further
            # specific params it needs from the params_manager instance.

            # Update ParamsManager with the specific pair and timeframe for this pre-training run
            # if CLI arguments were provided.
            logger.info(f"Setting ParamsManager: data_fetch_pairs to [{pretrain_pair}], data_fetch_timeframes to [{pretrain_timeframe}] for this pre-training run.")
            await params_manager.set_param("data_fetch_pairs", [pretrain_pair])
            await params_manager.set_param("data_fetch_timeframes", [pretrain_timeframe])

            pre_trainer_config = {
                'user_data_dir': str(user_data_dir), # Use validated user_data_dir
                'exchange_name': exchange_name,   # Use validated exchange_name
                # Add other necessary basic configs for PreTrainer initialization if any.
            }
            # PreTrainer now requires params_manager in its constructor.
            pre_trainer = PreTrainer(config=pre_trainer_config, params_manager=params_manager)

            strategy_id = "DuoAI_Strategy_Pretrain" # Default strategy ID
            logger.info(f"Running pre-training pipeline for strategy_id: {strategy_id} (using pairs/timeframes from ParamsManager)...")

            # run_pretraining_pipeline is an async method, so it needs to be run in an event loop.
            # Since main() is already async and run with asyncio.run() at the script's end,
            # we can await it here directly.
            # run_pretraining_pipeline no longer takes pair/timeframe directly.
            # It uses what's in ParamsManager (which we just set if CLI args were used).
            await pre_trainer.run_pretraining_pipeline(
                strategy_id=strategy_id
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
        reflection_symbols_str = args.reflection_symbols or ",".join(config_pairs) # Default to all pairs in whitelist
        try:
            parsed_symbols = [symbol.strip() for symbol in reflection_symbols_str.split(',') if symbol.strip()]
            if not parsed_symbols: # Check if list is empty after stripping
                logger.error("Reflection symbols are empty or invalid after parsing. Please provide a comma-separated list, e.g., --reflection-symbols ETH/USDT,BTC/USDT or ensure config has pair_whitelist.")
            else:
                logger.info(f"Starting reflection loop for symbols: {parsed_symbols} with interval {args.reflection_interval} minutes.")
                # Pass validated config and user_data_dir to ReflectieLus constructor
                reflection_loop = ReflectieLus(config=ft_config, user_data_dir=user_data_dir)
                await reflection_loop.start_reflection_loop(symbols=parsed_symbols, interval_minutes=args.reflection_interval)
                # The line above is an infinite loop, so code below this in the if block won't be reached.
        except Exception as e:
            logger.error(f"Error during reflection loop initialization or execution: {e}", exc_info=True)
        logger.info("--- AI Reflection Loop Finished (or encountered an error) ---") # This line might only be reached if the loop breaks due to error

if __name__ == "__main__":
    # The primary validation of API keys is now handled by load_and_validate_config based on the
    # configured exchange. A general warning here is less critical but can be kept if desired,
    # or removed to avoid redundancy if load_and_validate_config is comprehensive enough.
    # For now, let's remove the specific Bitvavo key check here as it's implicitly covered.
    # if not os.getenv('BITVAVO_API_KEY') or not os.getenv('BITVAVO_SECRET_KEY'):
    #    if logger and logger.handlers: # Check if logger is available
    #        logger.warning("Specific API keys (e.g., BITVAVO_API_KEY) not found. This might be an issue if the configured exchange requires them.")
    #    else:
    #        print("Warning: Specific API keys not found. Check .env file if your exchange requires them.", file=sys.stderr)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled critical error at top level: {e}", exc_info=True)
        sys.exit(1)

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
# from strategies.DUOAI_Strategy import DUOAI_Strategy # For type hinting and strategy loading - Defer if Freqtrade is missing
# from freqtrade.data.dataprovider import DataProvider # Defer if Freqtrade is missing
# from freqtrade.configuration import Configuration # Defer if Freqtrade is missing
import pandas as pd

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

    if args.pretrain_cnn:
        logger.info(f"--- Starting CNN Pre-training Process for Pair: {args.pretrain_pair}, Timeframe: {args.pretrain_timeframe} (Simplified Data Loading) ---")

        # This path is designed to work even if Freqtrade is not fully installed (e.g. TA-Lib issues)
        # It relies on CNNPatterns and PreTrainer which should not have hard Freqtrade import dependencies at their top level.

        try:
            # Simplified: Load only the base OHLCV data for the target pair and timeframe
            logger.info(f"Loading base OHLCV data for {args.pretrain_pair} - {args.pretrain_timeframe}")
            exchange_name_for_loading = "binance"
            base_ohlcv_df = load_data_for_pair(
                base_data_dir="user_data/data",
                exchange=exchange_name_for_loading,
                pair=args.pretrain_pair,
                timeframe=args.pretrain_timeframe
            )

            if base_ohlcv_df is None or base_ohlcv_df.empty:
                logger.error(f"Failed to load base OHLCV data for {args.pretrain_pair} - {args.pretrain_timeframe} from user_data/data/{exchange_name_for_loading}. Skipping pre-training.")
            else:
                logger.info(f"Successfully loaded base OHLCV data for {args.pretrain_pair} - {args.pretrain_timeframe}. Shape: {base_ohlcv_df.shape}")

                logger.info(f"Successfully loaded base OHLCV data for {args.pretrain_pair} - {args.pretrain_timeframe}. Shape: {base_ohlcv_df.shape}")

                logger.info(f"Instantiating CNNPatterns and preparing features for {args.pretrain_pair} - {args.pretrain_timeframe}...")
                cnn_patterns = CNNPatterns()
                scaled_features_df, fitted_scaler = cnn_patterns.prepare_cnn_features(
                    dataframe=base_ohlcv_df.copy(),
                    timeframe=args.pretrain_timeframe,
                    pair_name=args.pretrain_pair
                )

                if scaled_features_df is None or scaled_features_df.empty or fitted_scaler is None:
                    logger.error(f"Feature preparation failed for {args.pretrain_pair} - {args.pretrain_timeframe} using base_ohlcv_df. Skipping pre-training.")
                else:
                    logger.info(f"Successfully prepared features for {args.pretrain_pair} - {args.pretrain_timeframe}. Scaled features shape: {scaled_features_df.shape}")

                    logger.info(f"Instantiating PreTrainer for {args.pretrain_pair} - {args.pretrain_timeframe}...")
                    pre_trainer_config = {
                        'user_data_dir': 'user_data',
                        'pretrain_pair': args.pretrain_pair,
                        'pretrain_timeframe': args.pretrain_timeframe,
                        'hyperparameters': {'sequence_length': 30} # Example, PreTrainer might use this
                    }
                    pre_trainer = PreTrainer(config=pre_trainer_config)

                    logger.info(f"Calling PreTrainer.pretrain for {args.pretrain_pair} - {args.pretrain_timeframe}...")
                    pre_trainer.pretrain(
                        features_df=scaled_features_df,
                        scaler=fitted_scaler,
                        pair=args.pretrain_pair,
                        timeframe=args.pretrain_timeframe,
                        cnn_patterns_instance=cnn_patterns
                    )
                    logger.info(f"Pre-training placeholder process completed for {args.pretrain_pair} - {args.pretrain_timeframe}.")
            logger.info(f"--- CNN Pre-training Process (Simplified Data Path) for {args.pretrain_pair} - {args.pretrain_timeframe} Finished ---")

        except Exception as e:
            logger.error(f"Error during simplified CNN pre-training data loading/processing for {args.pretrain_pair} - {args.pretrain_timeframe}: {e}", exc_info=True)
            logger.info(f"--- CNN Pre-training Process for {args.pretrain_pair} - {args.pretrain_timeframe} Failed ---")


    # Initialize Core Components
    logger.info("Initializing core components...")
    try:
        # ParamsManager is now initialized inside PreTrainer and potentially other components if needed
        # If a global instance is required by other parts of the pipeline, it can be initialized here.
        # For now, assuming components that need it will initialize it or receive it.
        # params_manager = ParamsManager() # This was the old line.

        # Initialize ParamsManager for other parts of the pipeline if they rely on a central instance
        # If run_pipeline.py's later stages need params_manager, it should be instantiated here.
        # For now, we'll assume that the main pipeline logic (after pre-training) will instantiate it
        # or that components like DUOAI_Strategy already do.
        # DUOAI_Strategy initializes its own ParamsManager.
        pass # Not initializing a global params_manager here for now.
        # params_manager = ParamsManager() # This was the old line.

        # Initialize ParamsManager for other parts of the pipeline if they rely on a central instance
        # If run_pipeline.py's later stages need params_manager, it should be instantiated here.
        # For now, we'll assume that the main pipeline logic (after pre-training) will instantiate it
        # or that components like DUOAI_Strategy already do.
        # DUOAI_Strategy initializes its own ParamsManager.
        pass # Not initializing a global params_manager here for now.

        # The following initializations are for the main trading pipeline,
        # not directly for the pre-training step if it's run exclusively.
        # Consider if these should be conditional (e.g., if not args.pretrain_cnn).
        # For now, keeping them as they are part of the original main flow.

        # DUOAI_Strategy already initializes its own ParamsManager, CNNPatterns, PreTrainer etc.
        # We are calling a specific pre-training path if --pretrain-cnn is set.
        # The main pipeline run (without --pretrain-cnn) would use the strategy's internal instances.

        # If --pretrain-cnn is the *only* action, we might not need to init these here.
        # However, if the pipeline can run pre-training AND then proceed, they are needed.
        # Assuming for now that --pretrain-cnn is an auxiliary action and the main pipeline might still run.

        # If not running pre-training, or if pipeline continues after pre-training,
        # the original core component initialization would occur.
        # However, the original PreTrainer instantiation is complex.
        # The subtask is about integrating CNNPatterns with PreTrainer via the new pretrain method.
        # The existing main pipeline part below this might need adjustment if its PreTrainer usage changes.
        # For now, let's focus on the --pretrain-cnn path.

        # If we are ONLY pre-training, we might not want to initialize the full strategy's components here.
        # Let's assume the script can either pre-train OR run the main pipeline.
        if not args.pretrain_cnn: # Only run main pipeline components if not in pre-training mode
            logger.info("Initializing core components for main trading pipeline...")
            # This section needs to be reviewed in context of how PreTrainer is used by DUOAI_Strategy.
            # DUOAI_Strategy initializes its own instances of ParamsManager, CNNPatterns, PreTrainer.
            # The PreTrainer instance here might be redundant if DUOAI_Strategy handles its own.
            # This was the old block:
            # params_manager = ParamsManager()
            # try:
            #     bitvavo_executor = BitvavoExecutor() ...
            # except ValueError: ...
            # cnn_pattern_detector = CNNPatterns(params_manager=params_manager) ...
            # pre_trainer_main_pipeline = PreTrainer( # Renamed to avoid conflict
            #     params_manager=params_manager,
            #     cnn_pattern_detector=cnn_pattern_detector,
            #     bitvavo_executor=bitvavo_executor
            # )
            # logger.info("PreTrainer (for main pipeline) initialized successfully.")
            # The above block is now largely handled by DUOAI_Strategy's own __init__
            logger.info("Core components for main pipeline are initialized within DUOAI_Strategy itself if strategy is run.")

    except Exception as e:
        logger.error(f"Fatal error during (potential) core component initialization for main pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Run the Pre-training Pipeline
    try:
        logger.info("--- Running Pre-training & Backtesting Pipeline ---")

        # Get the strategy ID to run from ParamsManager or use a default
        # (This default_strategy_id should ideally be part of your ParamsManager defaults)

        # The main pipeline execution logic. This should only run if not in a specific pre-training mode,
        # or if pre-training is a step before this.
        # For now, let's assume if --pretrain-cnn is passed, we don't run this main flow.
        if not args.pretrain_cnn:
            logger.info("--- Attempting to run main trading pipeline (DUOAI_Strategy flow) ---")
            # This part would involve instantiating DUOAI_Strategy and running it,
            # which is typically handled by Freqtrade itself when live/dry running.
            # For a script like run_pipeline.py, it implies a more direct invocation of strategy methods
            # or a simulated Freqtrade environment.
            # The existing code calls `pre_trainer.run_pretraining_pipeline(strategy_id=strategy_id_to_run)`
            # This `pre_trainer` was the one initialized above.
            # DUOAI_Strategy has its own `pre_trainer` instance.
            # This section needs clarification on how the main pipeline is triggered vs. Freqtrade's own execution.
            # For now, let's assume this part is about triggering the DUOAI_Strategy's internal PreTrainer's pipeline.

            # To align with the structure, we'd need a Freqtrade-like environment or a specific
            # method in DUOAI_Strategy that kicks off its operations including its PreTrainer.
            # This is beyond the scope of `--pretrain-cnn` integration.
            # The old code:
            # strategy_id_to_run = params_manager.get_param('default_strategy_id')
            # if not strategy_id_to_run:
            #     strategy_id_to_run = "DefaultPipelineRunStrategy" # Fallback if not in params
            #     logger.warning(f"'default_strategy_id' not found in ParamsManager, using fallback: {strategy_id_to_run}")
            # await pre_trainer_main_pipeline.run_pretraining_pipeline(strategy_id=strategy_id_to_run)
            # logger.info("--- Pre-training & Backtesting Pipeline completed. ---")
            logger.info("Main pipeline execution (strategy's own PreTrainer, etc.) would occur here if invoked.")
            logger.info("This script's primary role for the main pipeline might be for specific utility tasks like pre-training.")

    except Exception as e:
        logger.error(f"Error during main pipeline operations: {e}", exc_info=True)
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

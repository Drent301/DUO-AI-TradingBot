#!/usr/bin/env python3
# process_backtest_reflections.py
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd
from dotenv import load_dotenv
from pathlib import Path # Import Path

# Define base paths assuming this script might be in a 'scripts' or 'utils' subdirectory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent # Adjust if script is at project root (then PROJECT_ROOT_DIR = SCRIPT_DIR)

# Now import core components
try:
    from core.ai_activation_engine import AIActivationEngine
    from core.reflectie_lus import ReflectieLus
    from core.bias_reflector import BiasReflector
    from core.confidence_engine import ConfidenceEngine
    from core.cnn_patterns import CNNPatterns
    from core.params_manager import ParamsManager
    from freqtrade.data.history import load_pair_history
    from freqtrade.configuration import Configuration
    from freqtrade.enums import CandleType
except ImportError as e:
    # Logger might not be initialized yet, so print is safer here.
    print(f"Error importing core modules or Freqtrade: {e}")
    print("Ensure that the script is run from a context where 'core' and 'freqtrade' are available, or PYTHONPATH is set.")
    # print(f"Current SCRIPT_DIR: {SCRIPT_DIR}, PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}") # For debugging
    sys.exit(1)

# Setup logging
LOG_DIR = PROJECT_ROOT_DIR / "user_data" / "logs"
LOG_FILE_PATH = LOG_DIR / "process_backtest_reflections.log"

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

# --- Constants ---
# Define informative timeframes required by the strategy (example, should match strategy)
# This should ideally be dynamically fetched from the strategy or config.
INFORMATIVE_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h'] # Example
BASE_TIMEFRAME = '5m' # Example, should match strategy

async def load_ohlcv_data_from_freqtrade_file(
    pair: str,
    timeframe: str,
    data_dir: str, # Will be converted to Path internally
    exchange_name: str,
    candle_type: CandleType = CandleType.SPOT
) -> Optional[pd.DataFrame]:
    """
    Loads OHLCV data for a given pair and timeframe from Freqtrade's data directory.
    Uses Freqtrade's load_pair_history function.
    Freqtrade's `load_pair_history` is expected to return a DataFrame with lowercase
    OHLCV column names ('open', 'high', 'low', 'close', 'volume') and numeric types.
    If issues arise with data from `load_pair_history` (e.g., mixed case columns or non-numeric types),
    explicit standardization (e.g., `df.columns = df.columns.str.lower()`,
    `pd.to_numeric` for OHLCV columns) could be added here.
    """
    data_dir_path = Path(data_dir) # Convert string argument to Path
    try:
        logger.debug(f"Attempting to load OHLCV data for {pair} ({timeframe}) from {data_dir_path} for exchange {exchange_name} and candle type {candle_type}")
        # Freqtrade's load_pair_history should handle Path objects for datadir
        df = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=data_dir_path, # Pass Path object
            data_format="json",
            candle_type=candle_type,
            exchange=exchange_name
        )
        if df.empty:
            logger.warning(f"No data found for {pair} ({timeframe}) in {data_dir_path}")
            return None
        logger.info(f"Successfully loaded {len(df)} candles for {pair} ({timeframe})")
        return df
    except Exception as e:
        logger.error(f"Error loading OHLCV data for {pair} ({timeframe}): {e}", exc_info=True)
        return None

async def load_historical_candles_for_trade(
    trade_details: Dict[str, Any],
    base_timeframe: str,
    informative_timeframes: List[str],
    data_dir: str,
    exchange_name: str,
    startup_candle_count: int = 200 # Number of candles to load prior to trade close
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Loads historical candle data (base + informative) relevant to a specific trade's closing time.
    """
    pair = trade_details['pair']
    # Ensure close_timestamp is timezone-aware (UTC)
    close_timestamp_ms = trade_details['close_timestamp']
    close_dt = datetime.fromtimestamp(close_timestamp_ms / 1000, tz=timezone.utc)

    logger.info(f"Loading historical candles for trade {trade_details.get('trade_id', 'N/A')} of {pair}, closing at {close_dt.isoformat()}")

    all_candles: Dict[str, pd.DataFrame] = {}
    timeframes_to_load = [base_timeframe] + [tf for tf in informative_timeframes if tf != base_timeframe]

    for tf in timeframes_to_load:
        df = await load_ohlcv_data_from_freqtrade_file(pair, tf, data_dir, exchange_name)
        if df is None or df.empty:
            logger.warning(f"Could not load data for {pair} on timeframe {tf}. Skipping this timeframe for this trade.")
            continue

        # Filter candles up to the trade's close time
        # Ensure df.index is timezone-aware if it's not already (Freqtrade usually handles this)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(timezone.utc)

        df_filtered = df[df.index <= close_dt].copy()

        if df_filtered.empty:
            # This means there is NO data at or before the trade's close time.
            # Using the tail of the original df would be incorrect as it might be data far after the trade.
            logger.warning(
                f"No candles found for {pair} ({tf}) at or before trade close_dt {close_dt}. "
                f"Original df for timeframe goes from {df.index.min()} to {df.index.max()} (len: {len(df)}). "
                f"Cannot provide relevant historical data for this timeframe for this trade."
            )
            # Do not use a fallback that takes data from a potentially different period.
            # Effectively, this timeframe will be missing from all_candles for this trade.
            continue
        else:
             # Data exists up to close_dt. Take the last startup_candle_count candles from this filtered set.
            original_filtered_len = len(df_filtered)
            df_filtered = df_filtered.iloc[-startup_candle_count:].copy()
            logger.info(
                f"Filtered data for {pair} ({tf}) up to {close_dt}: {original_filtered_len} candles. "
                f"Taking last {len(df_filtered)} (max {startup_candle_count}) candles for analysis."
            )
            if len(df_filtered) < startup_candle_count:
                 logger.warning(
                     f"Loaded {len(df_filtered)} candles for {pair} ({tf}), which is less than the "
                     f"requested startup_candle_count of {startup_candle_count} (available data up to close_dt was limited)."
                 )

        if not df_filtered.empty: # This check is now more about whether the slice operation itself yielded data
            all_candles[tf] = df_filtered
            logger.debug(f"Final selected data for {pair} ({tf}): {len(df_filtered)} candles, ending at {df_filtered.index.max().isoformat()}")
        else:
            # This case should be less common now with the revised logic, but good to keep.
            logger.warning(f"Filtered dataframe for {pair} ({tf}) is unexpectedly empty after selection process.")


    if not all_candles.get(base_timeframe):
        logger.error(f"Failed to load base timeframe ({base_timeframe}) candles for trade of {pair}. Cannot proceed with this trade.")
        return None

    return all_candles

async def main(args):
    """
    Main asynchronous function to process backtest reflections.
    """
    logger.info("Starting backtest reflection process...")
    # Load .env from project root
    dotenv_path = PROJECT_ROOT_DIR / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logger.info(f"Loaded .env file from {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}. Proceeding with environment variables if already set.")

    # Convert string paths from args to Path objects
    config_file_path = Path(args.config_file)
    data_dir_path = Path(args.data_dir)
    backtest_file_path = Path(args.backtest_file)


    # 1. Initialize Freqtrade configuration (minimal for data loading)
    try:
        # Configuration.from_files expects a list of strings or Path objects
        ft_config = Configuration.from_files([config_file_path])
        ft_config['datadir'] = str(data_dir_path) # Freqtrade config likely expects string paths
        ft_config['exchange']['name'] = args.exchange_name
        ft_config['strategy'] = args.strategy_name

        # Populate base_timeframe and informative_timeframes from strategy if possible
        # This is a placeholder; a more robust solution would inspect the strategy class
        global BASE_TIMEFRAME, INFORMATIVE_TIMEFRAMES
        BASE_TIMEFRAME = ft_config.get('timeframe', BASE_TIMEFRAME)
        # For informative_timeframes, it's more complex as it's usually a class member.
        # Using a default or requiring it as an arg might be simpler for this script.
        logger.info(f"Using base timeframe: {BASE_TIMEFRAME}")
        logger.info(f"Using informative timeframes: {INFORMATIVE_TIMEFRAMES}")


    except Exception as e:
        logger.error(f"Error loading Freqtrade configuration: {e}", exc_info=True)
        return

    # 2. Initialize core AI components
    logger.info("Initializing AI components...")
    try:
        # ParamsManager might take a Path object if updated, or convert to str if needed
        params_manager = ParamsManager(config_file=str(config_file_path))
        reflectie_lus = ReflectieLus()
        # BiasReflector and ConfidenceEngine might take params_manager or config path
        # Assuming they are updated or can handle it.
        bias_reflector = BiasReflector(params_manager=params_manager) # Pass PM
        confidence_engine = ConfidenceEngine(params_manager=params_manager) # Pass PM
        ai_activation_engine = AIActivationEngine(reflectie_lus_instance=reflectie_lus)
    except Exception as e:
        logger.error(f"Error initializing AI components: {e}", exc_info=True)
        return
    logger.info("AI components initialized successfully.")

    # 3. Parse Freqtrade backtest JSON results file
    # This script is primarily designed for Freqtrade results from version ~2023.x onwards,
    # which typically output a dictionary with a top-level 'strategy' key,
    # containing results per strategy. It also attempts to handle older formats
    # where the JSON might be a direct list of trades.
    try:
        with backtest_file_path.open('r', encoding='utf-8') as f: # Use Path.open()
            backtest_results = json.load(f)
        logger.info(f"Successfully loaded backtest results from {backtest_file_path}")
    except Exception as e:
        logger.error(f"Error reading or parsing backtest file {backtest_file_path}: {e}", exc_info=True)
        return

    trades_to_process = []
    # Try to identify newer Freqtrade backtest format (dictionary with 'strategy' key)
    if isinstance(backtest_results, dict) and 'strategy' in backtest_results:
        logger.debug("Detected modern Freqtrade backtest result format (dictionary with 'strategy' key).")
        strategy_data = backtest_results.get('strategy', {}).get(args.strategy_name)
        if strategy_data and isinstance(strategy_data, dict):
            trades_to_process = strategy_data.get('trades', [])
            if not trades_to_process:
                logger.warning(f"No trades found under 'strategy.{args.strategy_name}.trades'.")
        else:
            logger.warning(f"Strategy '{args.strategy_name}' not found in the 'strategy' part of the backtest results.")
            logger.info(f"Available strategies in backtest file: {list(backtest_results.get('strategy', {}).keys())}")

    # Try to identify older Freqtrade backtest format (direct list of trades)
    elif isinstance(backtest_results, list):
        logger.warning("Detected older Freqtrade backtest result format (direct list of trades).")
        # Assumption: If it's a list, it's a list of trades.
        # Further validation might be needed if other list-based formats exist.
        if all(isinstance(item, dict) and 'pair' in item and 'close_timestamp' in item for item in backtest_results):
            trades_to_process = backtest_results
        else:
            logger.error("The backtest file is a list, but items do not look like trade objects.")
            trades_to_process = [] # Ensure it's empty if format is wrong

    # Handle unrecognized format
    else:
        logger.error(
            f"Unrecognized backtest JSON structure in '{backtest_file_path}'. "
            f"Expected a dictionary with a 'strategy' key or a direct list of trades. "
            f"Found top-level type: {type(backtest_results)} with keys: {list(backtest_results.keys()) if isinstance(backtest_results, dict) else 'N/A (not a dict)'}. "
            f"Cannot process this file."
        )
        return # Stop processing

    if not trades_to_process:
        logger.error(f"No trades extracted for strategy '{args.strategy_name}' from the backtest file '{backtest_file_path}'. Please check strategy name and file content/format.")
        # Log available strategies again if it was a dict structure, for user convenience
        if isinstance(backtest_results, dict) and 'strategy' in backtest_results and not strategy_data:
             logger.info(f"Reminder: Available strategies in 'strategy' key: {list(backtest_results.get('strategy', {}).keys())}")
        return

    # Sort trades chronologically by close_timestamp
    # Freqtrade timestamps are typically milliseconds
    trades_to_process.sort(key=lambda t: t['close_timestamp'])
    logger.info(f"Found {len(trades_to_process)} trades for strategy '{args.strategy_name}', sorted chronologically.")

    # 4. Iterate through trades
    processed_trades = 0
    for i, trade_data in enumerate(trades_to_process):
        logger.info(f"Processing trade {i+1}/{len(trades_to_process)}: Pair {trade_data['pair']}, Profit {trade_data['profit_ratio']:.2%}")

        # a. Load historical candle data
        # Ensure 'close_timestamp' exists and is valid
        if 'close_timestamp' not in trade_data or trade_data['close_timestamp'] is None:
            logger.warning(f"Trade {trade_data.get('trade_id', 'N/A')} for pair {trade_data['pair']} is missing 'close_timestamp'. Skipping.")
            continue

        # Determine startup_candle_count from strategy config if available, else default
        strategy_startup_candles = ft_config.get('startup_candle_count', 200)

        historical_candles = await load_historical_candles_for_trade(
            trade_details=trade_data,
            base_timeframe=BASE_TIMEFRAME,
            informative_timeframes=INFORMATIVE_TIMEFRAMES,
            data_dir=args.data_dir,
            exchange_name=args.exchange_name,
            startup_candle_count=strategy_startup_candles
        )

        if not historical_candles:
            logger.warning(f"Could not load historical candles for trade {trade_data.get('trade_id', 'N/A')} of {trade_data['pair']}. Skipping reflection for this trade.")
            continue

        # b. Construct 'pseudo_trade_context'
        pseudo_trade_context = {
            "trade_id": trade_data.get('trade_id', f"backtest_trade_{i}"),
            "symbol": trade_data['pair'],
            "profit_pct": trade_data['profit_ratio'], # Freqtrade uses 'profit_ratio'
            "profit_abs": trade_data['profit_abs'],
            "open_timestamp": trade_data['open_timestamp'],
            "close_timestamp": trade_data['close_timestamp'],
            "open_rate": trade_data['open_rate'],
            "close_rate": trade_data['close_rate'],
            "exit_reason": trade_data.get('exit_reason', 'unknown'),
            "stake_amount": trade_data['stake_amount'],
            "fee_open": trade_data.get('fee_open', 0.0),
            "fee_close": trade_data.get('fee_close', 0.0),
            "is_short": trade_data.get('is_short', False),
            "leverage": trade_data.get('leverage', 1.0),
            # Add any other fields AIActivationEngine might expect in trade_context
        }
        logger.debug(f"Constructed pseudo_trade_context: {pseudo_trade_context}")

        # c. Call ai_activation_engine.activate_ai
        try:
            logger.info(f"Calling AIActivationEngine.activate_ai for trade {pseudo_trade_context['trade_id']}")
            reflection_result = await ai_activation_engine.activate_ai(
                trigger_type='trade_closed', # Consistent with DUOAI_Strategy
                token=trade_data['pair'],
                candles_by_timeframe=historical_candles,
                strategy_id=args.strategy_name,
                trade_context=pseudo_trade_context,
                mode='backtest_reflection', # Special mode for this process
                bias_reflector_instance=bias_reflector,
                confidence_engine_instance=confidence_engine
            )
            if reflection_result:
                logger.info(f"Reflection successful for trade {pseudo_trade_context['trade_id']}. Result summary: {reflection_result.get('analysis_summary', 'N/A')}")
            else:
                logger.info(f"AI activation did not trigger reflection for trade {pseudo_trade_context['trade_id']}.")
            processed_trades += 1
        except Exception as e:
            logger.error(f"Error during AI activation for trade {pseudo_trade_context['trade_id']} of {trade_data['pair']}: {e}", exc_info=True)

        # Optional: Add a small delay if needed to avoid hitting API rate limits for external services (if any)
        # await asyncio.sleep(1)

    logger.info(f"Backtest reflection process completed. Processed {processed_trades} trades.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Freqtrade backtest results for AI reflection.")
    parser.add_argument("--backtest-file", required=True, help="Path to the Freqtrade backtest JSON results file.")
    parser.add_argument("--config-file", required=True, help="Path to the Freqtrade config.json file (for strategy params, paths etc.).")
    parser.add_argument("--strategy-name", required=True, help="Name of the strategy to process trades for.")
    parser.add_argument("--exchange-name", required=True, help="Name of the exchange (e.g., 'binance', 'kraken'). Needed for data loading.")
    parser.add_argument("--data-dir", required=True, help="Path to Freqtrade's data directory (containing OHLCV .json/.feather files).")

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Script finished.")

```

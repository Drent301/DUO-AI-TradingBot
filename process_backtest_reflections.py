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

# Modify sys.path to ensure 'core' modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR) # This should be the project root
sys.path.insert(0, PARENT_DIR)

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
    print(f"Error importing core modules or Freqtrade: {e}")
    print("Ensure that the script is run from a context where 'core' and 'freqtrade' are available.")
    print(f"Current SCRIPT_DIR: {SCRIPT_DIR}, PARENT_DIR: {PARENT_DIR}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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
    data_dir: str,
    exchange_name: str,
    candle_type: CandleType = CandleType.SPOT
) -> Optional[pd.DataFrame]:
    """
    Loads OHLCV data for a given pair and timeframe from Freqtrade's data directory.
    Uses Freqtrade's load_pair_history function.
    """
    try:
        logger.debug(f"Attempting to load OHLCV data for {pair} ({timeframe}) from {data_dir} for exchange {exchange_name} and candle type {candle_type}")
        # Freqtrade's load_pair_history handles both .json and .feather
        df = load_pair_history(
            pair=pair,
            timeframe=timeframe,
            datadir=data_dir,
            data_format="json", # Or detect based on files available; Freqtrade handles this
            candle_type=candle_type,
            exchange=exchange_name # Required by Freqtrade
        )
        if df.empty:
            logger.warning(f"No data found for {pair} ({timeframe}) in {data_dir}")
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
            logger.warning(f"No candles found for {pair} ({tf}) before or at {close_dt}. Raw df length: {len(df)}")
            # Try to get the last N candles if no candles are before close_dt (e.g. if trade is very old)
            if not df.empty:
                 df_filtered = df.iloc[-startup_candle_count:].copy()
                 logger.info(f"Using last {startup_candle_count} candles for {pair} ({tf}) as a fallback.")
            else:
                logger.warning(f"Still no candles for {pair} ({tf}) after fallback.")
                continue
        else:
             # Ensure sufficient history (e.g., last N candles up to close_dt)
            df_filtered = df_filtered.iloc[-startup_candle_count:].copy()
            if len(df_filtered) < startup_candle_count:
                 logger.warning(f"Loaded {len(df_filtered)} candles for {pair} ({tf}), less than requested {startup_candle_count}")

        if not df_filtered.empty:
            all_candles[tf] = df_filtered
            logger.debug(f"Loaded {len(df_filtered)} candles for {pair} ({tf}) up to {close_dt.isoformat()}")
        else:
            logger.warning(f"Filtered dataframe for {pair} ({tf}) is empty after attempting to get {startup_candle_count} candles.")


    if not all_candles.get(base_timeframe):
        logger.error(f"Failed to load base timeframe ({base_timeframe}) candles for trade of {pair}. Cannot proceed with this trade.")
        return None

    return all_candles

async def main(args):
    """
    Main asynchronous function to process backtest reflections.
    """
    logger.info("Starting backtest reflection process...")
    load_dotenv(os.path.join(PARENT_DIR, '.env')) # Load .env from project root

    # 1. Initialize Freqtrade configuration (minimal for data loading)
    try:
        ft_config = Configuration.from_files([args.config_file])
        ft_config['datadir'] = args.data_dir # Override datadir from CLI
        ft_config['exchange']['name'] = args.exchange_name # Ensure exchange name is set
        # Set strategy name if needed by any Freqtrade components used
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
        params_manager = ParamsManager(config_file=args.config_file) # Assumes config.json structure
        reflectie_lus = ReflectieLus() # Dependencies like PromptBuilder are initialized internally
        bias_reflector = BiasReflector(params_manager=params_manager)
        confidence_engine = ConfidenceEngine(params_manager=params_manager)
        # cnn_patterns = CNNPatterns() # Initialized within ReflectieLus's PromptBuilder or AIActivationEngine
        ai_activation_engine = AIActivationEngine(reflectie_lus_instance=reflectie_lus)
    except Exception as e:
        logger.error(f"Error initializing AI components: {e}", exc_info=True)
        return
    logger.info("AI components initialized successfully.")

    # 3. Parse Freqtrade backtest JSON results file
    try:
        with open(args.backtest_file, 'r') as f:
            backtest_results = json.load(f)
        logger.info(f"Successfully loaded backtest results from {args.backtest_file}")
    except Exception as e:
        logger.error(f"Error reading or parsing backtest file {args.backtest_file}: {e}", exc_info=True)
        return

    # Extract trades - structure depends on Freqtrade version
    # Assuming results are a list of strategies, each with a list of trades
    trades_to_process = []
    if isinstance(backtest_results, dict) and 'strategy' in backtest_results: # Newer format
        strategy_results = backtest_results.get('strategy', {}).get(args.strategy_name)
        if strategy_results:
            trades_to_process = strategy_results.get('trades', [])
    elif isinstance(backtest_results, list): # Older format might be a list of trades directly
        # This needs careful checking of the actual JSON structure from your Freqtrade version
        logger.warning("Processing older backtest format (list of trades). Ensure this matches your file structure.")
        trades_to_process = backtest_results

    if not trades_to_process:
        logger.error(f"No trades found for strategy '{args.strategy_name}' in the backtest file, or file format not recognized as expected.")
        if isinstance(backtest_results, dict) and 'strategy' in backtest_results :
             logger.info(f"Available strategies in backtest file: {list(backtest_results['strategy'].keys())}")
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

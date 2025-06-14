# core/ai_optimizer.py
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta # Added timedelta
from typing import List, Dict, Any, Optional
import pandas as pd # Added pandas
import sqlite3 # Added sqlite3
# import dotenv # No longer needed as __main__ is removed
# import sys # No longer needed as __main__ is removed

# Attempt to import necessary components from other core modules
try:
    from core.pre_trainer import PreTrainer
    from core.strategy_manager import StrategyManager
    # These functions are assumed to be in reflectie_analyser.py
    from core.reflectie_analyser import analyse_reflecties, generate_mutation_proposal, analyze_timeframe_bias
    from core.market_data_provider import get_recent_market_data # Added import
    from core.pattern_performance_analyzer import PatternPerformanceAnalyzer # Added import
    from core.pattern_weight_optimizer import PatternWeightOptimizer # Added import
except ImportError as e:
    logging.warning(f"AIOptimizer: Error importing dependency: {e}. Some features may not work.")
    # Define placeholders if imports fail, to allow basic structure testing
    PreTrainer = None
    StrategyManager = None
    def analyse_reflecties(*args, **kwargs): return {}
    def generate_mutation_proposal(*args, **kwargs): return None
    def analyze_timeframe_bias(*args, **kwargs): return 0.5


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define paths for logging and reflection data
# LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs') # Old path
# os.makedirs(LOG_DIR, exist_ok=True) # Old path
# OPTIMIZER_LOG_FILE = os.path.join(LOG_DIR, 'optimizer_activity_log.json') # Old path
# REFLECTION_LOG_FILE = os.path.join(LOG_DIR, 'reflectie-logboek.json') # Old path, to be updated by new main

OPTIMIZER_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory', 'ai_optimizer_log.json')
os.makedirs(os.path.dirname(OPTIMIZER_LOG_FILE), exist_ok=True)

FREQTRADE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'freqtrade.sqlite')
# Note: REFLECTION_LOG_FILE will be defined in the new __main__ block's context or used by analyse_reflecties directly.

# Default strategy ID to work with
DEFAULT_STRATEGY_ID = "DUOAI_Strategy"

class AIOptimizer:
    """
    Orchestrates AI-driven optimization cycles, including pre-training,
    strategy performance analysis, reflection, mutation, and pattern weight tuning.
    """
    def __init__(self, pre_trainer: PreTrainer, strategy_manager: StrategyManager):
        """
        Initializes the AIOptimizer.

        Args:
            pre_trainer: An instance of PreTrainer for model pre-training.
            strategy_manager: An instance of StrategyManager for managing strategy parameters
                              and performance. This manager's ParamsManager is also used by
                              PatternPerformanceAnalyzer and PatternWeightOptimizer.
        """
        if pre_trainer is None or strategy_manager is None:
            logger.error("AIOptimizer could not be initialized: PreTrainer or StrategyManager is None.")
            # Or raise ValueError, as PreTrainer/StrategyManager types are expected from type hints
            raise ValueError("PreTrainer and StrategyManager instances are required for AIOptimizer.")

        self.pre_trainer = pre_trainer
        self.strategy_manager = strategy_manager

        # Instantiate PatternPerformanceAnalyzer and PatternWeightOptimizer
        # Use the params_manager from strategy_manager for consistency.
        # Pass None for paths to let PatternPerformanceAnalyzer use its defaults or fetch from params_manager.
        try:
            self.pattern_analyzer = PatternPerformanceAnalyzer(
                params_manager=self.strategy_manager.params_manager,
                freqtrade_db_path=None, # Let PPA handle default/param fetching
                pattern_log_path=None   # Let PPA handle default/param fetching
            )
            self.pattern_weight_optimizer = PatternWeightOptimizer(
                params_manager=self.strategy_manager.params_manager,
                pattern_performance_analyzer=self.pattern_analyzer
            )
            logger.info("PatternPerformanceAnalyzer and PatternWeightOptimizer initialized.")
        except Exception as e:
            logger.error(f"Error initializing PPA or PWO in AIOptimizer: {e}", exc_info=True)
            # Decide if these are critical. For now, log and continue, optimizers might fail later.
            self.pattern_analyzer = None
            self.pattern_weight_optimizer = None

        logger.info("AIOptimizer initialized.")

    def _log_optimization_activity(self, activity_type: str, details: Dict[str, Any]):
        """Logs optimization activities to a structured JSON log file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "details": details
        }
        try:
            with open(OPTIMIZER_LOG_FILE, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2)
                f.write('\n')
        except IOError as e:
            logger.error(f"Error writing to optimizer log: {e}")

    async def _get_all_closed_trades_from_db(self) -> pd.DataFrame:
        # Method content as per issue description
        conn = None
        try:
            conn = await asyncio.to_thread(sqlite3.connect, FREQTRADE_DB_PATH)
            query = "SELECT * FROM trades WHERE is_open = 0;"
            df = pd.read_sql_query(query, conn)
            df['open_date'] = pd.to_datetime(df['open_date'])
            df['close_date'] = pd.to_datetime(df['close_date'])
            return df
        except sqlite3.Error as e:
            logger.error(f"Databasefout bij ophalen trades: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Onverwachte fout bij ophalen trades: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    async def _learn_preferred_pairs(self):
        # Method content as per issue description
        logger.info("[AIOptimizer] Leren van preferredPairs...")
        all_trades_df = await self._get_all_closed_trades_from_db()

        if all_trades_df.empty:
            logger.info("Geen gesloten trades gevonden om preferredPairs te leren.")
            return

        recent_trades_df = all_trades_df[all_trades_df['close_date'] > (datetime.now() - timedelta(days=30))]

        if recent_trades_df.empty:
            logger.info("Geen recente trades gevonden voor preferredPairs. Sla leren over.")
            return

        pair_performance = recent_trades_df.groupby('pair')['profit_pct'].agg(['mean', 'count'])
        pair_performance = pair_performance[pair_performance['count'] >= 5]

        preferred_pairs_df = pair_performance.sort_values(by='mean', ascending=False)

        # Fetch top_n_pairs from ParamsManager
        top_n_pairs = self.strategy_manager.params_manager.get_param("preferredPairsCount", default=5)
        logger.info(f"[AIOptimizer] Using preferredPairsCount: {top_n_pairs} (Default: 5)")

        preferred_pairs = preferred_pairs_df.head(top_n_pairs).index.tolist()

        if preferred_pairs:
            logger.info(f"Geleerde preferredPairs (top {top_n_pairs}): {preferred_pairs}")
            await self.strategy_manager.params_manager.set_param("preferredPairs", preferred_pairs, strategy_id=None)
            await self._log_optimization_activity("learn_preferred_pairs", {"preferredPairs": preferred_pairs})
        else:
            logger.info("Geen preferredPairs bepaald op basis van recente prestaties.")
            await self.strategy_manager.params_manager.set_param("preferredPairs", [], strategy_id=None)

    async def run_periodic_optimization(self, symbols: List[str], timeframes: List[str]):
        """
        Runs a periodic optimization cycle:
        1. Pre-trains models for each symbol and timeframe. (Commented out for current focus)
        2. Analyzes reflections and current strategy performance for each symbol/timeframe.
        3. Proposes and applies mutations to strategy parameters if deemed beneficial.
        4. Optimizes CNN pattern weights based on overall performance.
        5. Learns and updates preferred trading pairs.
        """
        logger.info("Starting periodic optimization cycle...")
        self._log_optimization_activity("cycle_start", {"symbols": symbols, "timeframes": timeframes})

        # In a real scenario, pre-training might be more targeted or less frequent.
        # await self.pre_trainer.run_pretraining_pipeline(symbols, timeframes)
        # self._log_optimization_activity("pre_training_completed", {})
        # For now, focusing on performance analysis and mutation part for this task

        strategy_id = DEFAULT_STRATEGY_ID

        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Optimizing for {symbol} on timeframe {timeframe} using strategy {strategy_id}")

                # 1. Get current strategy performance from the database
                current_strategy_performance = self.strategy_manager.get_strategy_performance(strategy_id)
                logger.info(f"Current performance for {strategy_id} ({symbol}, {timeframe}): {current_strategy_performance}")
                self._log_optimization_activity("performance_fetched", {
                    "strategy_id": strategy_id, "symbol": symbol, "timeframe": timeframe,
                    "performance": current_strategy_performance
                })

                # 2. Analyze reflections (external data from REFLECTION_LOG_FILE)
                # analyse_reflecties is assumed to load its data from REFLECTION_LOG_FILE
                reflection_insights = analyse_reflecties(symbol=symbol, timeframe=timeframe, strategy_id=strategy_id)

                # 3. Analyze timeframe bias (placeholder for more complex logic)
                # This might involve specific metrics or models not detailed here.
                timeframe_bias_score = analyze_timeframe_bias(symbol, timeframe, strategy_id)

                # 4. Get current strategy parameters from ParamsManager via StrategyManager
                # StrategyManager's ParamsManager is the source of truth for parameters.
                current_strategy_parameters = await self.strategy_manager.params_manager.get_param("strategies", strategy_id=strategy_id)
                if not current_strategy_parameters:
                    current_strategy_parameters = {} # Default if no params found
                    logger.warning(f"No parameters found for {strategy_id} via ParamsManager. Using empty params.")

                # Construct current_strategy_info for mutation proposal
                # It needs parameters and performance. Other metadata might be useful.
                current_strategy_info_for_mutation = {
                    "parameters": current_strategy_parameters,
                    "performance": current_strategy_performance,
                    # "last_mutated": (fetch from params_manager or strategy_manager if available/needed)
                    # "mutation_rationale": (fetch likewise)
                }

                # 4.5 Fetch recent market data for context (NEW STEP)
                logger.info(f"Fetching recent market data for {symbol} - {timeframe} for optimization context...")
                market_data_df = await get_recent_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    exchange="binance", # Assuming default exchange, make configurable if needed
                    days_to_load=90,    # Assuming default days, make configurable if needed
                    download_if_missing=True # Allow download
                )

                if market_data_df is None:
                    logger.warning(f"Could not fetch market data for {symbol} - {timeframe}. Proceeding without it for mutation proposal.")
                    # Decide if this is critical. For now, we'll allow generate_mutation_proposal to receive None.
                else:
                    logger.info(f"Successfully fetched market data for {symbol} - {timeframe}. Shape: {market_data_df.shape}")

                self._log_optimization_activity("market_data_fetched", {
                    "strategy_id": strategy_id, "symbol": symbol, "timeframe": timeframe,
                    "data_fetched": market_data_df is not None,
                    "shape": market_data_df.shape if market_data_df is not None else None
                })

                # 5. Generate mutation proposal
                # Note: generate_mutation_proposal will need to be updated to accept market_data_df
                mutation_proposal = generate_mutation_proposal(
                    insights=reflection_insights,
                    current_performance=current_strategy_performance,
                    current_strategy_info=current_strategy_info_for_mutation, # Pass synthesized info
                    symbol=symbol,
                    timeframe=timeframe,
                    timeframe_bias=timeframe_bias_score,
                    strategy_id=strategy_id,
                    market_data_df=market_data_df # Pass market data
                )

                if mutation_proposal:
                    logger.info(f"Mutation proposed for {strategy_id}: {mutation_proposal}")
                    self._log_optimization_activity("mutation_proposed", {
                        "strategy_id": strategy_id, "proposal": mutation_proposal
                    })
                    # 6. Apply mutation if proposal is valid
                    success = await self.strategy_manager.mutate_strategy(strategy_id, mutation_proposal)
                    if success:
                        logger.info(f"Strategy {strategy_id} mutated successfully.")
                        self._log_optimization_activity("mutation_applied", {"strategy_id": strategy_id, "success": True})
                    else:
                        logger.warning(f"Failed to mutate strategy {strategy_id}.")
                        self._log_optimization_activity("mutation_failed", {"strategy_id": strategy_id, "success": False})
                else:
                    logger.info(f"No mutation proposed for {strategy_id} for {symbol} on {timeframe}.")
                    self._log_optimization_activity("no_mutation_proposed", {
                        "strategy_id": strategy_id, "symbol": symbol, "timeframe": timeframe
                    })

        # --- CNN Pattern Weight Optimization ---
        # This step runs once per cycle, after individual symbol/timeframe analysis & mutations,
        # to adjust pattern weights based on their overall collected performance.
        if self.pattern_weight_optimizer:
            logger.info("Attempting to optimize CNN pattern weights...")
            try:
                # PatternWeightOptimizer.optimize_pattern_weights is a synchronous method.
                # It's run in a separate thread to avoid blocking the asyncio event loop.
                weights_optimized = await asyncio.to_thread(self.pattern_weight_optimizer.optimize_pattern_weights)
                if weights_optimized:
                    logger.info("CNN pattern weights optimized successfully.")
                    self._log_optimization_activity("pattern_weights_optimized", {"success": True})
                else:
                    logger.info("CNN pattern weights optimization did not result in changes or encountered an issue.")
                    self._log_optimization_activity("pattern_weights_optimization_no_change", {"success": False})
            except Exception as e:
                logger.error(f"Error during CNN pattern weight optimization: {e}", exc_info=True)
                self._log_optimization_activity("pattern_weights_optimization_error", {"error": str(e)})
        else:
            logger.warning("PatternWeightOptimizer not available. Skipping pattern weight optimization.")

        await self._learn_preferred_pairs() # Added call
        logger.info("Periodic optimization cycle finished.") # Ensure this is the final log msg as per instr.
        self._log_optimization_activity("cycle_end", {})

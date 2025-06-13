# core/ai_optimizer.py
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta # Added timedelta
from typing import List, Dict, Any, Optional
import pandas as pd # Added pandas
import sqlite3 # Added sqlite3

# Attempt to import necessary components from other core modules
try:
    from core.pre_trainer import PreTrainer
    from core.strategy_manager import StrategyManager
    # These functions are assumed to be in reflectie_analyser.py
    from core.reflectie_analyser import analyse_reflecties, generate_mutation_proposal, analyze_timeframe_bias
except ImportError as e:
    logging.warning(f"AIOptimizer: Error importing dependency: {e}. Some features may not work.")
    # Define placeholders if imports fail, to allow basic structure testing
    PreTrainer = None
    StrategyManager = None
    def analyse_reflecties(*args: Any, **kwargs: Any) -> Dict[str, Any]: return {}
    def generate_mutation_proposal(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]: return None
    def analyze_timeframe_bias(*args: Any, **kwargs: Any) -> float: return 0.5


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
    strategy performance analysis, reflection, and mutation.
    """
    def __init__(self):
        if PreTrainer is None or StrategyManager is None:
            logger.error("AIOptimizer could not be initialized due to missing PreTrainer or StrategyManager.")
            raise ImportError("PreTrainer or StrategyManager not available.")

        self.pre_trainer = PreTrainer()
        self.strategy_manager = StrategyManager() # Assumes StrategyManager initializes its own ParamsManager
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
        1. Pre-trains models for each symbol and timeframe.
        2. Analyzes reflections and current strategy performance.
        3. Proposes and applies mutations if deemed beneficial.
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

                # 5. Generate mutation proposal
                mutation_proposal = generate_mutation_proposal(
                    insights=reflection_insights,
                    current_performance=current_strategy_performance,
                    current_strategy_info=current_strategy_info_for_mutation, # Pass synthesized info
                    symbol=symbol,
                    timeframe=timeframe,
                    timeframe_bias=timeframe_bias_score,
                    strategy_id=strategy_id
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

        await self._learn_preferred_pairs() # Added call
        logger.info("Periodic optimization cycle finished.") # Ensure this is the final log msg as per instr.
        self._log_optimization_activity("cycle_end", {})


if __name__ == "__main__":
    import dotenv
    import sys
    # Note: pandas, sqlite3, datetime, timedelta should be imported at the top of the file.
    # Ensure these are covered by step 1.

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    # Setup a dummy Freqtrade database with some trades for testing
    # FREQTRADE_DB_PATH is now a global constant
    if not os.path.exists(FREQTRADE_DB_PATH):
        print(f"Waarschuwing: Freqtrade database {FREQTRADE_DB_PATH} niet gevonden. Maak een dummy DB aan.")
        conn_test = sqlite3.connect(FREQTRADE_DB_PATH)
        cursor = conn_test.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                pair TEXT NOT NULL,
                strategy TEXT NOT NULL,
                profit_abs REAL,
                profit_pct REAL,
                is_open INTEGER,
                open_date TIMESTAMP,
                close_date TIMESTAMP
            );
        """)
        dummy_trades_data = [
            # Original ETH trades (likely not recent enough or not DUOAI_Strategy)
            (1, "ETH/USDT", "DUOAI_Strategy", 10.0, 0.05, 0, datetime(2024,1,1).isoformat(), datetime(2024,1,2).isoformat()),
            (2, "ETH/USDT", "DUOAI_Strategy", -5.0, -0.025, 0, datetime(2024,1,3).isoformat(), datetime(2024,1,4).isoformat()),
            (3, "ETH/USDT", "DUOAI_Strategy", 8.0, 0.04, 0, datetime(2024,1,5).isoformat(), datetime(2024,1,6).isoformat()),
            # Original BTC trade (wrong strategy for preferredPairs logic if it filters by strategy, also not recent)
            (4, "BTC/USDT", "Another_Strategy", 20.0, 0.03, 0, datetime(2024,1,1).isoformat(), datetime(2024,1,2).isoformat()),

            # ZEN/USDT trades (target: 5 recent trades for "DUOAI_Strategy")
            # Original recent ZEN trades:
            (5, "ZEN/USDT", "DUOAI_Strategy", 15.0, 0.06, 0, (datetime.now() - timedelta(days=10)).isoformat(), (datetime.now() - timedelta(days=9)).isoformat()), # profit_pct = 0.06
            (6, "ZEN/USDT", "DUOAI_Strategy", 2.0, 0.01, 0, (datetime.now() - timedelta(days=5)).isoformat(), (datetime.now() - timedelta(days=4)).isoformat()),  # profit_pct = 0.01
            # Added ZEN trades to meet count >= 5 and ensure it's preferred:
            (8, "ZEN/USDT", "DUOAI_Strategy", 10.0, 0.05, 0, (datetime.now() - timedelta(days=12)).isoformat(), (datetime.now() - timedelta(days=11)).isoformat()), # profit_pct = 0.05
            (9, "ZEN/USDT", "DUOAI_Strategy", 12.0, 0.04, 0, (datetime.now() - timedelta(days=8)).isoformat(), (datetime.now() - timedelta(days=7)).isoformat()),  # profit_pct = 0.04
            (10, "ZEN/USDT", "DUOAI_Strategy", 9.0, 0.03, 0, (datetime.now() - timedelta(days=3)).isoformat(), (datetime.now() - timedelta(days=2)).isoformat()),   # profit_pct = 0.03

            # LSK/BTC trade (original, 1 recent trade for DUOAI_Strategy, won't meet count >= 5)
            (7, "LSK/BTC", "DUOAI_Strategy", 5.0, 0.02, 0, (datetime.now() - timedelta(days=2)).isoformat(), (datetime.now() - timedelta(days=1)).isoformat()), # profit_pct = 0.02

            # ADA/USDT trades (target: 5 recent trades for "DUOAI_Strategy", with lower avg profit than ZEN)
            (11, "ADA/USDT", "DUOAI_Strategy", 5.0, 0.010, 0, (datetime.now() - timedelta(days=15)).isoformat(), (datetime.now() - timedelta(days=14)).isoformat()), # profit_pct = 0.010
            (12, "ADA/USDT", "DUOAI_Strategy", 6.0, 0.012, 0, (datetime.now() - timedelta(days=13)).isoformat(), (datetime.now() - timedelta(days=12)).isoformat()), # profit_pct = 0.012
            (13, "ADA/USDT", "DUOAI_Strategy", -2.0, -0.004, 0, (datetime.now() - timedelta(days=11)).isoformat(), (datetime.now() - timedelta(days=10)).isoformat()),# profit_pct = -0.004
            (14, "ADA/USDT", "DUOAI_Strategy", 8.0, 0.015, 0, (datetime.now() - timedelta(days=9)).isoformat(), (datetime.now() - timedelta(days=8)).isoformat()),  # profit_pct = 0.015
            (15, "ADA/USDT", "DUOAI_Strategy", 7.0, 0.013, 0, (datetime.now() - timedelta(days=7)).isoformat(), (datetime.now() - timedelta(days=6)).isoformat())   # profit_pct = 0.013
        ]
        cursor.executemany("INSERT OR IGNORE INTO trades (id, pair, strategy, profit_abs, profit_pct, is_open, open_date, close_date) VALUES (?,?,?,?,?,?,?,?)", dummy_trades_data)
        conn_test.commit()
        conn_test.close()
        print(f"Dummy database met trades aangemaakt in {FREQTRADE_DB_PATH}")

    # Mock reflectie logboek voor testdoeleinden
    mock_log_data = [
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.8, "combined_bias_reported": 0.7,
         "current_learned_bias": 0.6, "current_learned_confidence": 0.7,
         "trade_context": {"timeframe": "1h", "profit_pct": 0.02}, "timestamp": "2025-06-11T10:00:00Z"},
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.6, "combined_bias_reported": 0.4,
         "current_learned_bias": 0.65, "current_learned_confidence": 0.75,
         "trade_context": {"timeframe": "1h", "profit_pct": -0.01}, "timestamp": "2025-06-11T11:00:00Z"},
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.9, "combined_bias_reported": 0.8,
         "current_learned_bias": 0.7, "current_learned_confidence": 0.8,
         "trade_context": {"timeframe": "1h", "profit_pct": 0.04}, "timestamp": "2025-06-11T12:00:00Z"},
    ]
    # Path for reflectie-logboek.json from issue's AIOptimizer code
    reflectie_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory', 'reflectie-logboek.json')
    os.makedirs(os.path.dirname(reflectie_log_path), exist_ok=True)
    with open(reflectie_log_path, 'w', encoding='utf-8') as f:
        json.dump(mock_log_data, f, indent=2)
    print(f"Mock reflectie logboek aangemaakt op: {reflectie_log_path}")

    async def run_test_ai_optimizer():
        optimizer = AIOptimizer()

        test_symbols = ["ETH/USDT", "ZEN/USDT", "LSK/BTC", "ADA/USDT"] # Added ADA
        test_timeframes = ["1h"]

        print("\n--- Test AIOptimizer: run_periodic_optimization (Default preferredPairsCount=5) ---")
        # Ensure default is used first if not set
        await optimizer.strategy_manager.params_manager.set_param("preferredPairsCount", None) # Clear to ensure default is tested
        await optimizer.run_periodic_optimization(test_symbols, test_timeframes)

        preferred_pairs_after_opt_default = await optimizer.strategy_manager.params_manager.get_param("preferredPairs")
        print(f"Geleerde preferredPairs na optimalisatie (Default Count): {preferred_pairs_after_opt_default}")
        assert len(preferred_pairs_after_opt_default) <= 5
        # Based on dummy data: ZEN (avg profit ~0.038), ADA (avg profit ~0.0092), ETH (no recent DUOAI), LSK (1 trade)
        # Expected default top 2: ZEN/USDT, ADA/USDT (if LSK is filtered by count)
        assert "ZEN/USDT" in preferred_pairs_after_opt_default
        assert "ADA/USDT" in preferred_pairs_after_opt_default


        print("\n--- Test AIOptimizer: run_periodic_optimization (Custom preferredPairsCount=1) ---")
        await optimizer.strategy_manager.params_manager.set_param("preferredPairsCount", 1)
        await optimizer.run_periodic_optimization(test_symbols, test_timeframes) # Rerun learning part

        preferred_pairs_after_opt_custom = await optimizer.strategy_manager.params_manager.get_param("preferredPairs")
        print(f"Geleerde preferredPairs na optimalisatie (Custom Count=1): {preferred_pairs_after_opt_custom}")
        assert len(preferred_pairs_after_opt_custom) == 1
        assert "ZEN/USDT" in preferred_pairs_after_opt_custom # ZEN has highest avg profit in dummy data

        print("\nOptimalisatiecyclus voltooid. Controleer de logs en memory-bestanden.")


    asyncio.run(run_test_ai_optimizer())

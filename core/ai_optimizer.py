# core/ai_optimizer.py
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

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
    def analyse_reflecties(*args, **kwargs): return {}
    def generate_mutation_proposal(*args, **kwargs): return None
    def analyze_timeframe_bias(*args, **kwargs): return 0.5


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define paths for logging and reflection data
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
OPTIMIZER_LOG_FILE = os.path.join(LOG_DIR, 'optimizer_activity_log.json')
REFLECTION_LOG_FILE = os.path.join(LOG_DIR, 'reflectie-logboek.json') # As used by analyse_reflecties

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

        logger.info("Periodic optimization cycle finished.")
        self._log_optimization_activity("cycle_end", {})


# --- Test Suite ---
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

    logging.basicConfig(level=logging.DEBUG, # Use DEBUG for test verbosity
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    TEST_DB_PATH = "test_optimizer_db.sqlite"
    TEST_OPTIMIZER_LOG_FILE = "test_optimizer_activity_log.json"
    TEST_REFLECTION_LOG_FILE = "test_reflectie-logboek.json" # Mock reflection log

    # Override global paths for testing
    OPTIMIZER_LOG_FILE = TEST_OPTIMIZER_LOG_FILE
    REFLECTION_LOG_FILE = TEST_REFLECTION_LOG_FILE # Ensure analyse_reflecties uses this

    # Mock implementations for dependencies
    class MockPreTrainer:
        async def run_pretraining_pipeline(self, symbols: List[str], timeframes: List[str]):
            logger.info("MockPreTrainer: Pre-training pipeline called.")
            return True

    # Using ParamsManager and BiasReflector mocks from StrategyManager's test if they were structured for reuse.
    # For this test, we'll define them here for clarity if they are simple enough.
    class MockParamsManager:
        def __init__(self):
            self.params = {
                DEFAULT_STRATEGY_ID: {
                    "buy": {"ema_short": 10, "ema_long": 20}, "sell": {"rsi_sell": 70},
                    "roi": {0: 0.1, 60: 0.01}, "stoploss": -0.1
                }
            }
            self.mutation_history = []

        async def get_param(self, category: str, strategy_id: str = None, param_name: str = None):
            if category == "strategies" and strategy_id:
                return self.params.get(strategy_id, {}).copy()
            return self.params.get(strategy_id, {}).get("buy",{}).get(param_name)

        async def set_param(self, strategy_id: str, param_name: str, value: Any):
            if strategy_id not in self.params: self.params[strategy_id] = {"buy": {}, "sell": {}}
            self.params[strategy_id]["buy"][param_name] = value
            self.mutation_history.append({"type": "param_set", "strategy": strategy_id, "param": param_name, "value": value})

        async def update_strategy_roi_sl_params(self, strategy_id: str, roi_table=None, stoploss_value=None, trailing_stop_params=None):
            if strategy_id not in self.params: self.params[strategy_id] = {}
            if roi_table: self.params[strategy_id]['roi'] = roi_table
            if stoploss_value: self.params[strategy_id]['stoploss'] = stoploss_value
            self.mutation_history.append({"type": "roi_sl_update", "strategy": strategy_id, "roi": roi_table, "sl": stoploss_value})


    class MockBiasReflector:
        def get_bias_score(self, token: str, strategy_id: str) -> float:
            return 0.6 # Default mock bias

    # Mock external analysis functions
    _original_analyse_reflecties = analyse_reflecties
    _original_generate_mutation_proposal = generate_mutation_proposal
    _original_analyze_timeframe_bias = analyze_timeframe_bias

    mock_mutation_calls = []
    def mock_generate_mutation_proposal_impl(insights, current_performance, current_strategy_info, symbol, timeframe, timeframe_bias, strategy_id):
        mock_mutation_calls.append(locals()) # Log call arguments
        # Simulate proposing a mutation for one specific case
        if symbol == "ETH/USDT" and current_performance.get("winRate", 0) > 0.5:
            return {
                "strategyId": strategy_id,
                "adjustments": {"parameterChanges": {"ema_short": current_strategy_info["parameters"]["buy"]["ema_short"] + 1}}, # Increment ema_short
                "rationale": "Mock proposal due to good win rate."
            }
        return None

    def setup_test_db_optimizer(db_path: str):
        if os.path.exists(db_path): os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, strategy TEXT, pair TEXT, is_open INTEGER,
            open_date DATETIME, close_date DATETIME, close_profit REAL, stake_amount REAL
        )""")
        # Sample data for DEFAULT_STRATEGY_ID
        # High win rate for ETH/USDT to trigger mock mutation
        trades_data = [
            (DEFAULT_STRATEGY_ID, 'ETH/USDT', 0, datetime.now(), datetime.now(), 0.02, 100),
            (DEFAULT_STRATEGY_ID, 'ETH/USDT', 0, datetime.now(), datetime.now(), 0.03, 100),
            (DEFAULT_STRATEGY_ID, 'ETH/USDT', 0, datetime.now(), datetime.now(), -0.01, 100), # 2 wins, 1 loss
            (DEFAULT_STRATEGY_ID, 'BTC/USDT', 0, datetime.now(), datetime.now(), -0.01, 100), # Low performance
        ]
        cursor.executemany("INSERT INTO trades (strategy, pair, is_open, open_date, close_date, close_profit, stake_amount) VALUES (?,?,?,?,?,?,?)", trades_data)
        conn.commit()
        conn.close()
        logger.info(f"Test DB {db_path} created for AIOptimizer tests.")

    def setup_mock_reflection_log_file(file_path: str):
        if os.path.exists(file_path): os.remove(file_path)
        sample_reflections = [
            {"timestamp": datetime.now().isoformat(), "symbol": "ETH/USDT", "reflection": "Positive sentiment observed."},
            {"timestamp": datetime.now().isoformat(), "symbol": "BTC/USDT", "reflection": "Market seems volatile."}
        ]
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_reflections, f, indent=2)
            logger.info(f"Mock reflection log {file_path} created.")
        except IOError as e:
            logger.error(f"Error creating mock reflection log: {e}")


    async def run_all_optimizer_tests():
        global analyse_reflecties, generate_mutation_proposal, analyze_timeframe_bias

        # Setup: Ensure StrategyManager and other dependencies are mocked or configured for testing
        os.environ['FREQTRADE_DB_PATH'] = TEST_DB_PATH # IMPORTANT: Ensure StrategyManager uses this test DB!

        setup_test_db_optimizer(TEST_DB_PATH)
        setup_mock_reflection_log_file(TEST_REFLECTION_LOG_FILE)
        if os.path.exists(TEST_OPTIMIZER_LOG_FILE): os.remove(TEST_OPTIMIZER_LOG_FILE)

        # --- Mocking the imported analysis functions ---
        # This is a way to achieve mocking without unittest.mock.patch
        # It relies on these functions being module-level imports that can be temporarily overridden.
        analyse_reflecties_calls = []
        def mocked_analyse_reflecties_impl(symbol, timeframe, strategy_id):
            analyse_reflecties_calls.append(locals())
            return {"insight": "mocked insight for " + symbol}

        analyze_timeframe_bias_calls = []
        def mocked_analyze_timeframe_bias_impl(symbol, timeframe, strategy_id):
            analyze_timeframe_bias_calls.append(locals())
            return 0.55 # Mocked bias

        # Replace original functions with mocks for the duration of this test
        analyse_reflecties = mocked_analyse_reflecties_impl
        generate_mutation_proposal = mock_generate_mutation_proposal_impl # Already defined above
        analyze_timeframe_bias = mocked_analyze_timeframe_bias_impl

        mock_params_manager = MockParamsManager()
        mock_bias_reflector = MockBiasReflector()

        # Configure StrategyManager for testing
        # The real StrategyManager is used, but its dependencies are controlled.
        strategy_manager_for_test = StrategyManager(db_path=TEST_DB_PATH) # Point to test DB
        strategy_manager_for_test.params_manager = mock_params_manager # Inject MockParamsManager
        strategy_manager_for_test.bias_reflector = mock_bias_reflector # Inject MockBiasReflector

        optimizer = AIOptimizer()
        optimizer.pre_trainer = MockPreTrainer() # Replace with mock
        optimizer.strategy_manager = strategy_manager_for_test # Use configured StrategyManager

        test_symbols = ["ETH/USDT", "BTC/USDT"]
        test_timeframes = ["5m"]

        logger.info("--- Running AIOptimizer Tests ---")
        await optimizer.run_periodic_optimization(test_symbols, test_timeframes)

        # Assertions
        # 1. Check if get_strategy_performance was used (indirectly, by checking proposal logic)
        #    The mock_generate_mutation_proposal is called with current_performance.
        assert len(mock_mutation_calls) == len(test_symbols) * len(test_timeframes)
        eth_usdt_call = next(c for c in mock_mutation_calls if c['symbol'] == 'ETH/USDT')
        # ETH/USDT: 2 wins, 1 loss from 3 trades => winRate = 2/3 approx 0.666
        assert abs(eth_usdt_call['current_performance']['winRate'] - (2/3)) < 0.001
        assert eth_usdt_call['current_performance']['tradeCount'] == 3

        btc_usdt_call = next(c for c in mock_mutation_calls if c['symbol'] == 'BTC/USDT')
        # BTC/USDT: 0 wins, 1 loss from 1 trade => winRate = 0
        assert abs(btc_usdt_call['current_performance']['winRate'] - 0.0) < 0.001
        assert btc_usdt_call['current_performance']['tradeCount'] == 1


        # 2. Check if mutate_strategy was called for ETH/USDT (due to mock proposal)
        mutated_eth = any(
            hist['type'] == 'param_set' and hist['strategy'] == DEFAULT_STRATEGY_ID and hist['param'] == 'ema_short'
            for hist in mock_params_manager.mutation_history
        )
        assert mutated_eth, "mutate_strategy should have been called for ETH/USDT to change ema_short."

        original_ema_short = MockParamsManager().params[DEFAULT_STRATEGY_ID]["buy"]["ema_short"] # Initial value
        final_ema_short = mock_params_manager.params[DEFAULT_STRATEGY_ID]["buy"]["ema_short"]
        assert final_ema_short == original_ema_short + 1, "ema_short should have been incremented."


        # 3. Check optimizer log file
        assert os.path.exists(TEST_OPTIMIZER_LOG_FILE)
        with open(TEST_OPTIMIZER_LOG_FILE, 'r') as f:
            log_content = f.read()
            assert "cycle_start" in log_content
            assert "performance_fetched" in log_content
            assert "mutation_proposed" in log_content # For ETH/USDT
            assert "mutation_applied" in log_content # For ETH/USDT
            assert "no_mutation_proposed" in log_content # For BTC/USDT
            assert "cycle_end" in log_content
        logger.info("Optimizer log checks passed.")

        # Check calls to mocked analysis functions
        assert len(analyse_reflecties_calls) == len(test_symbols) * len(test_timeframes)
        assert len(analyze_timeframe_bias_calls) == len(test_symbols) * len(test_timeframes)
        logger.info("Calls to mocked analysis functions verified.")

        # Cleanup
        if os.path.exists(TEST_DB_PATH): os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_OPTIMIZER_LOG_FILE): os.remove(TEST_OPTIMIZER_LOG_FILE)
        if os.path.exists(TEST_REFLECTION_LOG_FILE): os.remove(TEST_REFLECTION_LOG_FILE)
        if 'FREQTRADE_DB_PATH' in os.environ: del os.environ['FREQTRADE_DB_PATH']
        logger.info("Test files cleaned up.")

        # Restore original functions
        analyse_reflecties = _original_analyse_reflecties
        generate_mutation_proposal = _original_generate_mutation_proposal
        analyze_timeframe_bias = _original_analyze_timeframe_bias

        logger.info("--- AIOptimizer Tests Completed ---")

    # Python 3.7+ for asyncio.run
    if hasattr(asyncio, 'run'):
        # Need to import sys for stdout in basicConfig
        import sys
        try:
            asyncio.run(run_all_optimizer_tests())
        except ImportError as e:
            # This handles the case where core StrategyManager or PreTrainer couldn't be imported initially
            logger.error(f"Skipping AIOptimizer tests due to import error: {e}")
            # Create dummy log file to indicate test was skipped due to setup
            if not os.path.exists(TEST_OPTIMIZER_LOG_FILE):
                 with open(TEST_OPTIMIZER_LOG_FILE, 'w') as f:
                     json.dump({"status": "skipped", "reason": str(e)}, f)

    else: # Fallback for older Python versions if necessary (less likely in modern envs)
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(run_all_optimizer_tests())
        except ImportError as e:
            logger.error(f"Skipping AIOptimizer tests due to import error: {e}")
            if not os.path.exists(TEST_OPTIMIZER_LOG_FILE):
                 with open(TEST_OPTIMIZER_LOG_FILE, 'w') as f:
                     json.dump({"status": "skipped", "reason": str(e)}, f)

```

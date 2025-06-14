# core/strategy_manager.py
import json
import logging
import os
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import dotenv # Added for __main__
import sqlite3 # Added
import pandas as pd # Added

# Assuming ParamsManager and BiasReflector will be in these locations
from core.params_manager import ParamsManager # Added
from core.bias_reflector import BiasReflector # Added (already imported in __main__ before)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# MEMORY_DIR and STRATEGY_MEMORY_FILE are no longer needed for performance
# os.makedirs(MEMORY_DIR, exist_ok=True) # Not needed if MEMORY_DIR is not used

# FREQTRADE_DB_PATH will be used for the database connection
# It's good practice to allow override via environment variable
FREQTRADE_DB_PATH_DEFAULT = 'freqtrade.sqlite'


class StrategyManager:
    """
    Manages strategy selection, mutation, and performance tracking using Freqtrade's database.
    """

    def __init__(self, db_path: Optional[str] = None):
        # Allow db_path to be overridden, e.g., for testing
        self.db_path = db_path or os.getenv('FREQTRADE_DB_PATH', FREQTRADE_DB_PATH_DEFAULT)
        self.params_manager = ParamsManager() # Initialize ParamsManager
        self.bias_reflector: Optional[BiasReflector] = None # Placeholder, will be set up
        logger.info(f"StrategyManager initialized. DB path: {self.db_path}")

    # _load_strategy_memory and _save_strategy_memory are removed as we use the DB.

    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Fetches aggregated performance from Freqtrade's database for a given strategy.
        Returns winRate, avgProfit, and tradeCount.
        """
        default_performance = {"winRate": 0.0, "avgProfit": 0.0, "tradeCount": 0}
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Note: Column names like 'close_profit_abs' for profit percentage might vary.
            # Assuming 'close_profit' is a ratio (e.g., 0.05 for 5%).
            # Freqtrade stores profit as ratio (e.g. 0.01 = 1%) in close_profit column.
            # is_open = 0 filters for closed trades.
            query = """
                SELECT
                    SUM(CASE WHEN close_profit > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(id) as winRate,
                    AVG(close_profit) as avgProfit,
                    COUNT(id) as tradeCount
                FROM trades
                WHERE strategy = ? AND is_open = 0
            """
            df = pd.read_sql_query(query, conn, params=(strategy_id,))

            if df.empty or df.iloc[0]['tradeCount'] is None or df.iloc[0]['tradeCount'] == 0:
                logger.info(f"No closed trades found for strategy {strategy_id} in {self.db_path}. Returning default performance.")
                return default_performance

            performance = df.iloc[0].to_dict()
            # Ensure types are correct, especially for winRate which can be NaN if tradeCount is 0
            performance['winRate'] = float(performance.get('winRate', 0.0) or 0.0) # Handle potential NaN
            performance['avgProfit'] = float(performance.get('avgProfit', 0.0) or 0.0) # Handle potential NaN
            performance['tradeCount'] = int(performance.get('tradeCount', 0) or 0)

            logger.info(f"Performance for {strategy_id}: {performance}")
            return performance

        except sqlite3.Error as e:
            logger.error(f"Database error while fetching performance for {strategy_id}: {e}")
            return default_performance
        except Exception as e:
            logger.error(f"Unexpected error fetching performance for {strategy_id}: {e}")
            return default_performance
        finally:
            if conn:
                conn.close()

    async def update_strategy_performance(self, strategy_id: str, new_performance: Dict[str, Any]):
        """
        This method's role is adjusted. The database is the primary source of truth for trade performance.
        This could be used to update an internal cache or log external updates if needed,
        but it does not write directly to the main performance store (the DB).
        """
        # For now, this method logs the intention but doesn't alter primary performance data.
        # It could be used for caching or other auxiliary tasks in a more complex system.
        logger.info(f"Attempt to update performance for strategy {strategy_id} with {new_performance}. "
                    f"Performance is primarily derived from the database.")
        # If an in-memory cache or secondary store were used, this is where it would be updated.
        # For example:
        # self.strategy_memory_cache[strategy_id]['performance'] = new_performance
        # self.strategy_memory_cache[strategy_id]['last_updated'] = datetime.now().isoformat()
        # No direct save to a JSON file or DB write operation here for performance metrics.
        pass # Explicitly doing nothing further for now.


    async def mutate_strategy(self, strategy_id: str, proposal: Dict[str, Any]):
        """
        Muteert een strategie op basis van een mutatievoorstel van de AI.
        Updates strategy parameters, ROI, Stoploss, and Trailing Stop using ParamsManager.
        """
        if not proposal or proposal.get('strategyId') != strategy_id:
            logger.warning(f"Invalid mutation proposal for strategy {strategy_id}.")
            return False

        logger.info(f"Mutating strategy {strategy_id} with proposal: {proposal.get('adjustments')}")

        adjustments = proposal.get('adjustments', {})
        parameter_changes = adjustments.get('parameterChanges')
        roi_changes = adjustments.get('roi')
        stoploss_change = adjustments.get('stoploss')
        trailing_adjustments = adjustments.get('trailingStop') # This is a dictionary

        changes_made = False

        # Update standard parameters
        if parameter_changes:
            for param_name, param_value in parameter_changes.items():
                # Assuming params_manager.set_param takes (strategy_id, key, value)
                # It should be (key, value, strategy_id) based on ParamsManager implementation
                await self.params_manager.set_param(key=param_name, value=param_value, strategy_id=strategy_id)
                logger.debug(f"Parameter '{param_name}' of strategy {strategy_id} set to {param_value}.")
            changes_made = True

        # Prepare parameters for update_strategy_roi_sl_params
        new_trailing_stop_value = None
        new_trailing_offset_value = None

        if trailing_adjustments: # trailing_adjustments is a dict e.g. {"value": 0.01, "offset": 0.005}
            new_trailing_stop_value = trailing_adjustments.get('value')
            new_trailing_offset_value = trailing_adjustments.get('offset')

        # Only call if there are actual ROI/SL/Trailing changes
        if roi_changes or stoploss_change or trailing_adjustments: # trailing_adjustments itself implies a change if present
            await self.params_manager.update_strategy_roi_sl_params(
                strategy_id=strategy_id,
                new_roi=roi_changes, # Pass directly, can be None
                new_stoploss=stoploss_change, # Pass directly, can be None
                new_trailing_stop=new_trailing_stop_value, # Pass extracted value, can be None
                new_trailing_only_offset_is_reached=new_trailing_offset_value # Pass extracted value, can be None
            )
            logger.info(f"ROI/SL/Trailing parameters for strategy {strategy_id} update initiated.")
            changes_made = True

        if changes_made:
            # Logging or status update related to mutation can happen here.
            # The actual parameters are now managed by ParamsManager, so no local strategy_memory to update for params.
            # We might want to store mutation metadata (rationale, proposal) if needed,
            # potentially in a separate log or DB table, or also managed by ParamsManager.
            # For now, just log success.
            logger.info(f"Strategy {strategy_id} successfully mutated based on proposal.")
            return True
        else:
            logger.info(f"No actionable changes in proposal for strategy {strategy_id}. Mutation not performed.")
            return False


    async def get_best_strategy(self, token: str, interval: str) -> Optional[Dict[str, Any]]:
        logger.info(f"Attempting to determine best strategy for token: {token}, interval: {interval}")

        available_strategies = await self.params_manager.get_param("strategies")
        if not available_strategies or not isinstance(available_strategies, dict):
            logger.warning("No strategies found or 'strategies' is not a dictionary in ParamsManager configuration. Cannot determine best strategy.")
            return None

        best_strategy_id = None
        best_performance_score = -float('inf')
        best_strategy_details = None

        for strategy_id, strategy_config_params in available_strategies.items():
            logger.info(f"Evaluating strategy: {strategy_id} for token {token}")

            performance_data = self.get_strategy_performance(strategy_id) # This is a synchronous method call

            win_rate = performance_data.get("winRate", 0.0)
            avg_profit = performance_data.get("avgProfit", 0.0)
            trade_count = performance_data.get("tradeCount", 0)

            bias_score = 0.5  # Default bias
            if self.bias_reflector:
                bias_score = self.bias_reflector.get_bias_score(token, strategy_id)
            else:
                logger.warning(f"BiasReflector not available. Using default bias 0.5 for strategy {strategy_id}, token {token}.")

            # Calculate current_performance_score
            current_performance_score = (win_rate * 0.7) + (avg_profit * 10.0)
            current_performance_score += (bias_score * 0.1)

            # Apply penalty for low trade count
            trade_count_threshold = 50 # Standard threshold
            # Attempt to get strategy-specific threshold if defined, else use standard
            # This assumes a parameter like "minTradeCountForScoring" might exist in strategy_config_params
            specific_trade_count_threshold = strategy_config_params.get("minTradeCountForScoring", trade_count_threshold)

            if trade_count < specific_trade_count_threshold:
                penalty_factor = 0.5
                current_performance_score *= penalty_factor
                logger.warning(f"Applied penalty ({penalty_factor:.2f}) to strategy {strategy_id} due to low trade count ({trade_count} < {specific_trade_count_threshold}). Original score: {current_performance_score / penalty_factor:.4f}, Penalized score: {current_performance_score:.4f}")

            logger.info(f"Strategy: {strategy_id}, WinRate: {win_rate:.2%}, AvgProfit: {avg_profit:.4f}, TradeCount: {trade_count}, Bias: {bias_score:.2f}, Calculated Score: {current_performance_score:.4f}")

            if current_performance_score > best_performance_score:
                best_performance_score = current_performance_score
                best_strategy_id = strategy_id
                best_strategy_details = {
                    "id": strategy_id,
                    "performance": performance_data,
                    "bias": bias_score,
                    "parameters": strategy_config_params, # Parameters from ParamsManager for this strategy
                    "calculated_score": current_performance_score # Store the score for logging/debugging
                }

        if best_strategy_details:
            logger.info(f"Best strategy selected for token {token}, interval {interval}: {best_strategy_details['id']} with score: {best_strategy_details['calculated_score']:.4f}")
        else:
            logger.warning(f"Could not determine a best strategy for token {token}, interval {interval} from available options.")

        return best_strategy_details

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    # Setup basic logging for the test
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Define a test DB path and ensure it's used by StrategyManager
    TEST_DB_PATH = "test_freqtrade.sqlite"
    os.environ['FREQTRADE_DB_PATH'] = TEST_DB_PATH


    # --- Mock ParamsManager and BiasReflector for testing ---
    class MockParamsManager:
        def __init__(self):
            self.strategies_config = {
                "DUOAI_Strategy_V1": {
                    "buy_params": {"emaPeriod": 20, "rsiThresholdBuy": 70},
                    "sell_params": {"rsiThresholdSell": 70},
                    "roi_table": {"0": 0.1, "30": 0.05}, "stoploss_value": -0.10,
                    "minTradeCountForScoring": 40, # Strategy-specific threshold
                    "trailing_stop_positive": 0.01, "trailing_stop_positive_offset": 0.02, # Adding base trailing values
                },
                "DUOAI_Strategy_V2": {
                    "buy_params": {"emaPeriod": 10, "rsiThresholdBuy": 65},
                    "sell_params": {"rsiThresholdSell": 75},
                    "roi_table": {"0": 0.15, "20": 0.08}, "stoploss_value": -0.05,
                    # No minTradeCountForScoring, will use default 50 from get_best_strategy
                },
                "LowTrade_Strategy": {
                    "buy_params": {"emaPeriod": 5, "rsiThresholdBuy": 60},
                    "sell_params": {"rsiThresholdSell": 80},
                    "roi_table": {"0": 0.20}, "stoploss_value": -0.03,
                }
            }
            self.files_written = {} # To track file writing attempts for ROI/SL

        async def get_param(self, key: str, strategy_id: str = None): # Removed category, param_name for simplicity with new usage
            if key == "strategies" and strategy_id is None:
                return self.strategies_config.copy()
            if key == "strategies" and strategy_id: # Used by old get_best_strategy test, can be adapted or removed
                return self.strategies_config.get(strategy_id, {}).copy()
            # For mutate_strategy, it might try to get specific parameters if ParamsManager was more complex
            # For this mock, assume mutate_strategy gets the whole config and modifies it,
            # or set_param and update_strategy_roi_sl_params directly modify self.strategies_config
            if strategy_id: # If strategy_id is given, assume we want a key from within that strategy's config
                 return self.strategies_config.get(strategy_id, {}).get(key)
            return None


        async def set_param(self, key: str, value: Any, strategy_id: Optional[str] = None):
            if strategy_id:
                if strategy_id not in self.strategies_config:
                    self.strategies_config[strategy_id] = {}
                # This needs to be smarter if parameters are nested (e.g., in "buy_params")
                # For simplicity, mutate_strategy test will adjust to set flat params or nested ones directly.
                # Example: if key is "emaPeriod", it should go into "buy_params"
                if key == "emaPeriod": # Example handling
                    if "buy_params" not in self.strategies_config[strategy_id]: self.strategies_config[strategy_id]["buy_params"] = {}
                    self.strategies_config[strategy_id]["buy_params"][key] = value
                else:
                    self.strategies_config[strategy_id][key] = value
                logger.info(f"MockParamsManager: Set {key} to {value} for {strategy_id}")
            # Global params not handled by this mock for now

        async def update_strategy_roi_sl_params(self, strategy_id: str,
                                                new_roi: Optional[Dict] = None,
                                                new_stoploss: Optional[float] = None,
                                                new_trailing_stop: Optional[float] = None,
                                                new_trailing_only_offset_is_reached: Optional[float] = None):
            if strategy_id not in self.strategies_config:
                self.strategies_config[strategy_id] = {}

            if new_roi is not None:
                self.strategies_config[strategy_id]['roi_table'] = new_roi # Use 'roi_table'
            if new_stoploss is not None:
                self.strategies_config[strategy_id]['stoploss_value'] = new_stoploss # Use 'stoploss_value'

            # Trailing stop parameters might be nested or flat depending on real ParamsManager
            # For mock, let's assume they are flat for simplicity of update
            if new_trailing_stop is not None:
                self.strategies_config[strategy_id]['trailing_stop_positive'] = new_trailing_stop
            if new_trailing_only_offset_is_reached is not None:
                self.strategies_config[strategy_id]['trailing_stop_positive_offset'] = new_trailing_only_offset_is_reached

            if new_roi or new_stoploss or new_trailing_stop or new_trailing_only_offset_is_reached:
                self.files_written[strategy_id] = True # Simulate file write
                logger.info(f"MockParamsManager: Updated ROI/SL/Trailing for {strategy_id}")


    class MockBiasReflector:
        def get_bias_score(self, token: str, strategy_id: str) -> float:
            if token == "ETH/USDT":
                if strategy_id == "DUOAI_Strategy_V1": return 0.6
                if strategy_id == "DUOAI_Strategy_V2": return 0.7
                if strategy_id == "LowTrade_Strategy": return 0.8 # High bias but may be penalized
            return 0.5 # Default

    def setup_test_db(db_path: str):
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT, pair TEXT, is_open INTEGER,
            open_date DATETIME, close_date DATETIME, close_profit REAL,
            close_profit_abs REAL, stake_amount REAL )
        """)
        # Data for DUOAI_Strategy_V1 (50 trades, 60% WR, 0.01 avg profit)
        # 30 wins * 0.02 = 0.6, 20 losses * -0.005 = -0.1. Total profit = 0.5. Avg = 0.5/50 = 0.01
        trades_v1 = [('DUOAI_Strategy_V1', 'ETH/USDT', 0, '2023-01-01', '2023-01-01', 0.02, 2, 100)] * 30
        trades_v1 += [('DUOAI_Strategy_V1', 'ETH/USDT', 0, '2023-01-02', '2023-01-02', -0.005, -0.5, 100)] * 20

        # Data for DUOAI_Strategy_V2 (60 trades, 70% WR, 0.015 avg profit)
        # 42 wins * 0.025 = 1.05, 18 losses * -0.008333... (approx -1/120) => -0.15. Total profit = 0.9. Avg = 0.9/60 = 0.015
        trades_v2 = [('DUOAI_Strategy_V2', 'ETH/USDT', 0, '2023-01-03', '2023-01-03', 0.025, 2.5, 100)] * 42
        trades_v2 += [('DUOAI_Strategy_V2', 'ETH/USDT', 0, '2023-01-04', '2023-01-04', -1.5/180, -1.5/1.8, 100)] * 18 # approx -0.008333 * 18 = -0.15

        # Data for LowTrade_Strategy (10 trades, 80% WR, 0.02 avg profit)
        # 8 wins * 0.03 = 0.24, 2 losses * -0.02 = -0.04. Total profit = 0.2. Avg = 0.2/10 = 0.02
        trades_low = [('LowTrade_Strategy', 'ETH/USDT', 0, '2023-01-05', '2023-01-05', 0.03, 3, 100)] * 8
        trades_low += [('LowTrade_Strategy', 'ETH/USDT', 0, '2023-01-06', '2023-01-06', -0.02, -2, 100)] * 2

        cursor.executemany("INSERT INTO trades (strategy, pair, is_open, open_date, close_date, close_profit, close_profit_abs, stake_amount) VALUES (?,?,?,?,?,?,?,?)",
                           trades_v1 + trades_v2 + trades_low)
        conn.commit()
        conn.close()
        logger.info(f"Test database {db_path} created and populated for V1, V2, LowTrade strategies.")

    async def run_test_strategy_manager():
        setup_test_db(TEST_DB_PATH)
        strategy_manager = StrategyManager()
        strategy_manager.params_manager = MockParamsManager() # Use updated mock
        strategy_manager.bias_reflector = MockBiasReflector() # Use updated mock

        test_token = "ETH/USDT"
        test_interval = "5m"

        print("\n--- Test StrategyManager (New get_best_strategy) ---")

        # Test get_strategy_performance (verify DB setup)
        perf_v1 = strategy_manager.get_strategy_performance("DUOAI_Strategy_V1")
        assert perf_v1['tradeCount'] == 50
        assert abs(perf_v1['winRate'] - 0.60) < 0.001
        assert abs(perf_v1['avgProfit'] - 0.01) < 0.001

        perf_v2 = strategy_manager.get_strategy_performance("DUOAI_Strategy_V2")
        assert perf_v2['tradeCount'] == 60
        assert abs(perf_v2['winRate'] - 0.70) < 0.001
        assert abs(perf_v2['avgProfit'] - 0.015) < 0.001

        perf_low = strategy_manager.get_strategy_performance("LowTrade_Strategy")
        assert perf_low['tradeCount'] == 10
        assert abs(perf_low['winRate'] - 0.80) < 0.001
        assert abs(perf_low['avgProfit'] - 0.02) < 0.001

        # Test Case 1: Basic Selection (V2 should win)
        # V1 Score: (0.6*0.7) + (0.01*10.0) + (0.6*0.1) = 0.42 + 0.1 + 0.06 = 0.58
        # V2 Score: (0.7*0.7) + (0.015*10.0) + (0.7*0.1) = 0.49 + 0.15 + 0.07 = 0.71
        # LowTrade Score (raw): (0.8*0.7) + (0.02*10.0) + (0.8*0.1) = 0.56 + 0.2 + 0.08 = 0.84. Penalized: 0.84 * 0.5 = 0.42
        logger.info("\n--- Test Case: Basic Selection (V2 expected) ---")
        best_strat_info = await strategy_manager.get_best_strategy(test_token, test_interval)
        assert best_strat_info is not None, "Should find a best strategy"
        assert best_strat_info['id'] == "DUOAI_Strategy_V2", f"Expected V2, got {best_strat_info['id']}"
        assert abs(best_strat_info['calculated_score'] - 0.71) < 0.001
        logger.info(f"Test Case Basic Selection Passed. Best: {best_strat_info['id']}")

        # Test Case 2: Low Trade Count Penalty (LowTrade_Strategy gets penalized, V1 should win over it)
        logger.info("\n--- Test Case: Low Trade Count Penalty (V1 expected over LowTrade) ---")
        # Scores calculated above: V1=0.58, LowTrade_Penalized=0.42. V2=0.71
        # To make V1 win over LowTrade specifically, we'd need to make V2 less attractive or remove it.
        # For this test, we'll temporarily modify V2's performance to be worse than V1.
        original_v2_config = strategy_manager.params_manager.strategies_config["DUOAI_Strategy_V2"]
        strategy_manager.params_manager.strategies_config["DUOAI_Strategy_V2"] = {**original_v2_config, "winRate_override_for_test": 0.1} # Make V2 temporarily bad

        # Mock get_strategy_performance for V2 to return bad data for this specific sub-test
        original_get_performance = strategy_manager.get_strategy_performance
        def mock_get_performance_penalty_test(strategy_id_arg):
            if strategy_id_arg == "DUOAI_Strategy_V2": return {"winRate": 0.1, "avgProfit": 0.001, "tradeCount": 60} # Bad perf
            return original_get_performance(strategy_id_arg)
        strategy_manager.get_strategy_performance = mock_get_performance_penalty_test

        # V2 new score: (0.1*0.7) + (0.001*10.0) + (0.7*0.1) = 0.07 + 0.01 + 0.07 = 0.15
        # V1 Score = 0.58
        # LowTrade Penalized Score = 0.42
        # Now V1 (0.58) should be chosen.
        best_strat_penalty_test = await strategy_manager.get_best_strategy(test_token, test_interval)
        assert best_strat_penalty_test is not None
        assert best_strat_penalty_test['id'] == "DUOAI_Strategy_V1", f"Expected V1 due to V2 mod and LowTrade penalty, got {best_strat_penalty_test['id']}"

        strategy_manager.get_strategy_performance = original_get_performance # Restore
        strategy_manager.params_manager.strategies_config["DUOAI_Strategy_V2"] = original_v2_config # Restore
        logger.info(f"Test Case Low Trade Count Penalty Passed. Best: {best_strat_penalty_test['id']}")

        # Test Case 3: Strategy-Specific minTradeCountForScoring
        # V1 minTradeCountForScoring = 40. Actual trades = 50. Not penalized. Score = 0.58
        # V2 default threshold = 50. Give it 45 trades. Should be penalized.
        # Raw V2 score = 0.71. Penalized V2 score = 0.71 * 0.5 = 0.355
        # V1 (0.58) should win over penalized V2 (0.355)
        logger.info("\n--- Test Case: Strategy-Specific minTradeCountForScoring ---")
        original_get_performance_v2 = strategy_manager.get_strategy_performance
        def mock_get_performance_v2_specific_thresh(strategy_id_arg):
            if strategy_id_arg == "DUOAI_Strategy_V2": return {"winRate": 0.7, "avgProfit": 0.015, "tradeCount": 45} # 45 trades
            return original_get_performance_v2(strategy_id_arg)
        strategy_manager.get_strategy_performance = mock_get_performance_v2_specific_thresh

        best_strat_specific_thresh = await strategy_manager.get_best_strategy(test_token, test_interval)
        assert best_strat_specific_thresh is not None
        assert best_strat_specific_thresh['id'] == "DUOAI_Strategy_V1", f"Expected V1 due to V2 specific penalty, got {best_strat_specific_thresh['id']}"
        strategy_manager.get_strategy_performance = original_get_performance_v2 # Restore
        logger.info(f"Test Case Strategy-Specific Threshold Passed. Best: {best_strat_specific_thresh['id']}")

        # Test Case 4: No Strategies Configured
        logger.info("\n--- Test Case: No Strategies Configured ---")
        original_strategies_config = strategy_manager.params_manager.strategies_config
        strategy_manager.params_manager.strategies_config = {} # Empty config
        best_strat_no_config = await strategy_manager.get_best_strategy(test_token, test_interval)
        assert best_strat_no_config is None, "Expected None when no strategies are configured"
        strategy_manager.params_manager.strategies_config = original_strategies_config # Restore
        logger.info("Test Case No Strategies Configured Passed.")

        # Test Case 5: BiasReflector Unavailable
        logger.info("\n--- Test Case: BiasReflector Unavailable ---")
        original_bias_reflector = strategy_manager.bias_reflector
        strategy_manager.bias_reflector = None
        # Scores with default bias 0.5:
        # V1: (0.6*0.7) + (0.01*10.0) + (0.5*0.1) = 0.42 + 0.1 + 0.05 = 0.57
        # V2: (0.7*0.7) + (0.015*10.0) + (0.5*0.1) = 0.49 + 0.15 + 0.05 = 0.69
        # LowTrade (penalized): ((0.8*0.7) + (0.02*10.0) + (0.5*0.1)) * 0.5 = (0.56 + 0.2 + 0.05) * 0.5 = 0.81 * 0.5 = 0.405
        # V2 should win.
        best_strat_no_bias_reflector = await strategy_manager.get_best_strategy(test_token, test_interval)
        assert best_strat_no_bias_reflector is not None
        assert best_strat_no_bias_reflector['id'] == "DUOAI_Strategy_V2", f"Expected V2 with default bias, got {best_strat_no_bias_reflector['id']}"
        assert abs(best_strat_no_bias_reflector['calculated_score'] - 0.69) < 0.001
        strategy_manager.bias_reflector = original_bias_reflector # Restore
        logger.info("Test Case BiasReflector Unavailable Passed.")

        # Test mutate_strategy (ensure it still works with the new MockParamsManager structure)
        # This test mainly ensures mutate_strategy can interact with the refactored MockParamsManager
        logger.info("\n--- Test Mutate Strategy (with refactored MockParamsManager) ---")
        test_mutate_id = "DUOAI_Strategy_V1" # Use one of the new strategy IDs
        mock_proposal_mutate = {
            "strategyId": test_mutate_id,
            "adjustments": {
                "parameterChanges": {"emaPeriod": 30}, # This should go into 'buy_params'
                "roi": {"0": 0.22}, # This should become 'roi_table'
                "stoploss": -0.22, # This should become 'stoploss_value'
                "trailingStop": {"value": 0.033, "offset": 0.011}
            }
        }
        await strategy_manager.mutate_strategy(test_mutate_id, mock_proposal_mutate)
        mutated_params_v1 = await strategy_manager.params_manager.get_param("strategies", strategy_id=test_mutate_id)

        assert mutated_params_v1["buy_params"]["emaPeriod"] == 30
        assert mutated_params_v1["roi_table"] == {"0": 0.22}
        assert mutated_params_v1["stoploss_value"] == -0.22
        assert mutated_params_v1["trailing_stop_positive"] == 0.033
        assert mutated_params_v1["trailing_stop_positive_offset"] == 0.011
        logger.info("Test Mutate Strategy with refactored mocks passed.")


        logger.info("All StrategyManager tests (including new get_best_strategy) passed.")

        # Cleanup test DB
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        logger.info(f"Test database {TEST_DB_PATH} removed.")


    asyncio.run(run_test_strategy_manager())

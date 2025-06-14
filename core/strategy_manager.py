# core/strategy_manager.py
import json
import logging
import os
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp # Added for Freqtrade API notification
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
            await self._notify_freqtrade_of_parameter_changes(strategy_id) # Notify Freqtrade
            return True
        else:
            logger.info(f"No actionable changes in proposal for strategy {strategy_id}. Mutation not performed.")
            return False


    async def get_best_strategy(self, token: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Selects the best performing strategy for a given token and interval,
        considering performance from DB and parameters from ParamsManager.
        """
        # In a multi-strategy system, you'd iterate or query available strategies.
        # For this example, we assume "DUOAI_Strategy" is a key strategy.
        # This method might need to list available strategies from ParamsManager or a config.
        strategy_id = "DUOAI_Strategy" # Example strategy

        performance = self.get_strategy_performance(strategy_id) # Fetches from DB

        # Retrieve current parameters from ParamsManager
        # Assuming get_param can fetch the entire parameter set for a strategy
        # The structure of what get_param returns will depend on ParamsManager's design.
        # It might return buy/sell params, ROI, SL, etc.
        strategy_params = await self.params_manager.get_param("strategies", strategy_id=strategy_id) # Assuming async
        if not strategy_params:
            logger.warning(f"Parameters for strategy {strategy_id} not found via ParamsManager.")
            # Fallback or default parameters might be loaded here if appropriate
            # For now, returning None or minimal info if params are crucial and missing.
            # Or, ensure params_manager always returns a dict, possibly empty.
            strategy_params = {}


        bias = 0.5 # Default bias
        if self.bias_reflector:
            # BiasReflector might need strategy_id and potentially current params or performance
            bias = self.bias_reflector.get_bias_score(token, strategy_id)
        else:
            logger.warning("BiasReflector not available in StrategyManager. Using default bias 0.5.")

        return {
            "id": strategy_id,
            "performance": performance, # From DB
            "bias": bias,
            "parameters": strategy_params # From ParamsManager
        }

    async def _notify_freqtrade_of_parameter_changes(self, strategy_id: str):
        """
        Notifies Freqtrade of parameter changes by calling its API.
        Users should verify the specific Freqtrade API endpoint and payload requirements.
        A common endpoint for reloading configuration is /api/v1/reloadconfig.
        If a more granular update (e.g., for a specific strategy's parameters without a full reload)
        is available in your Freqtrade version, that would be preferable.
        """
        logger.info(f"Attempting to notify Freqtrade of parameter changes for strategy: {strategy_id}")

        freqtrade_url = os.getenv("FREQTRADE_API_URL", "http://localhost:8080") # Default if not set
        freqtrade_token = os.getenv("FREQTRADE_API_TOKEN", None) # Optional token

        # Example endpoint: /api/v1/reloadconfig reloads the entire configuration.
        # Freqtrade might offer more specific endpoints to update parts of a strategy or live parameters.
        # Consult your Freqtrade API documentation.
        reload_endpoint = "/api/v1/reloadconfig"
        # reload_endpoint = f"/api/v1/strategy/{strategy_id}/reload" # Hypothetical more specific endpoint

        if not freqtrade_url:
            logger.warning("FREQTRADE_API_URL is not set. Cannot notify Freqtrade of parameter changes.")
            return

        headers = {"Content-Type": "application/json"}
        if freqtrade_token:
            headers["Authorization"] = f"Bearer {freqtrade_token}"

        # For /reloadconfig, the payload is often empty or a simple directive.
        # For more specific strategy updates, the payload might need to contain the new parameters.
        payload = {}
        # Example for a hypothetical specific update:
        # payload = {"strategy_id": strategy_id, "action": "refresh_parameters"}

        api_full_url = f"{freqtrade_url.rstrip('/')}{reload_endpoint}"

        logger.info(f"Calling Freqtrade API: POST {api_full_url} for strategy {strategy_id}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_full_url, json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        logger.info(f"Freqtrade API call successful for {strategy_id}. Status: {response.status}, Response: {response_text}")
                    else:
                        logger.error(f"Freqtrade API call failed for {strategy_id}. Status: {response.status}, Response: {response_text}")
        except aiohttp.ClientConnectorError as e: # More specific connection error
            logger.error(f"Error connecting to Freqtrade API at {freqtrade_url} for {strategy_id}: {e}")
        except aiohttp.ClientError as e: # Catch other aiohttp client errors
            logger.error(f"AIOHTTP client error during Freqtrade API call for {strategy_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Freqtrade API call for {strategy_id}: {e}", exc_info=True)


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
            self.params = {
                "DUOAI_Strategy": {
                    "buy": {"emaPeriod": 20, "rsiThresholdBuy": 65},
                    "sell": {"rsiThresholdSell": 75},
                    "roi": {0: 0.1, 30: 0.05, 60: 0.01},
                    "stoploss": -0.10,
                    "trailing": {"enabled": True, "value": 0.02}
                },
                "OtherStrategy": {
                    "buy": {"emaPeriod": 50, "rsiThresholdBuy": 60},
                    "sell": {"rsiThresholdSell": 80},
                    "roi": {0: 0.2, 30: 0.15, 60: 0.05},
                    "stoploss": -0.15,
                    "trailing": {"enabled": False}
                }
            }
            self.files_written = {} # To track file writing attempts for ROI/SL

        async def get_param(self, category: str, strategy_id: str = None, param_name: str = None):
            if category == "strategies" and strategy_id:
                return self.params.get(strategy_id, {}).copy() # Return a copy
            elif strategy_id and param_name:
                return self.params.get(strategy_id, {}).get("buy", {}).get(param_name) or \
                       self.params.get(strategy_id, {}).get("sell", {}).get(param_name)
            return None

        async def set_param(self, strategy_id: str, param_name: str, value: Any):
            if strategy_id not in self.params:
                self.params[strategy_id] = {"buy": {}, "sell": {}}
            # Simplistic: assume it's a buy param if not obviously something else
            # A real ParamsManager would know where to put it (buy/sell/other sections)
            if "buy" not in self.params[strategy_id]: self.params[strategy_id]["buy"] = {}
            self.params[strategy_id]["buy"][param_name] = value
            logger.info(f"MockParamsManager: Set {param_name} to {value} for {strategy_id}")

        async def update_strategy_roi_sl_params(self, strategy_id: str,
                                                new_roi: Optional[Dict] = None,
                                                new_stoploss: Optional[float] = None, # Corrected param name
                                                new_trailing_stop: Optional[float] = None,
                                                new_trailing_only_offset_is_reached: Optional[float] = None):
            if strategy_id not in self.params:
                self.params[strategy_id] = {}

            if new_roi is not None: # Renamed from roi_table
                self.params[strategy_id]['roi'] = new_roi
                logger.info(f"MockParamsManager: Updated ROI for {strategy_id} to {new_roi}")
            if new_stoploss is not None: # Renamed from stoploss_value
                self.params[strategy_id]['stoploss'] = new_stoploss
                logger.info(f"MockParamsManager: Updated stoploss for {strategy_id} to {new_stoploss}")

            # Ensure 'trailing' dict exists before trying to set sub-keys
            if 'trailing' not in self.params[strategy_id]:
                self.params[strategy_id]['trailing'] = {} # Initialize if not present

            if new_trailing_stop is not None:
                self.params[strategy_id]['trailing']['value'] = new_trailing_stop # Store under 'value'
                logger.info(f"MockParamsManager: Updated trailing_stop_positive for {strategy_id} to {new_trailing_stop}")

            if new_trailing_only_offset_is_reached is not None:
                self.params[strategy_id]['trailing']['offset'] = new_trailing_only_offset_is_reached # Store under 'offset'
                logger.info(f"MockParamsManager: Updated trailing_stop_positive_offset for {strategy_id} to {new_trailing_only_offset_is_reached}")

            # Simulate that ParamsManager would write these to a strategy file if any change was made
            if new_roi or new_stoploss or new_trailing_stop or new_trailing_only_offset_is_reached:
                self.files_written[strategy_id] = True


    class MockBiasReflector:
        def get_bias_score(self, token: str, strategy_id: str) -> float:
            # Return a predictable bias score for testing
            if token == "ETH/USDT" and strategy_id == "DUOAI_Strategy":
                return 0.75
            return 0.5

    def setup_test_db(db_path: str):
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT,
            pair TEXT,
            is_open INTEGER,
            open_date DATETIME,
            close_date DATETIME,
            close_profit REAL,
            close_profit_abs REAL, -- Assuming this is absolute profit value if needed
            stake_amount REAL
        )
        """)
        # Sample data for DUOAI_Strategy
        trades_data_duoai = [
            ('DUOAI_Strategy', 'ETH/USDT', 0, '2023-01-01 10:00:00', '2023-01-01 12:00:00', 0.02, 0.02*100, 100), # Win
            ('DUOAI_Strategy', 'ETH/USDT', 0, '2023-01-02 10:00:00', '2023-01-02 12:00:00', -0.01, -0.01*100, 100), # Loss
            ('DUOAI_Strategy', 'BTC/USDT', 0, '2023-01-03 10:00:00', '2023-01-03 12:00:00', 0.03, 0.03*100, 100), # Win
            ('DUOAI_Strategy', 'ETH/USDT', 1, '2023-01-04 10:00:00', None, None, None, 100), # Open trade
        ]
        # Sample data for OtherStrategy
        trades_data_other = [
            ('OtherStrategy', 'LTC/USDT', 0, '2023-01-05 10:00:00', '2023-01-05 12:00:00', 0.05, 0.05*200, 200), # Win
            ('OtherStrategy', 'LTC/USDT', 0, '2023-01-06 10:00:00', '2023-01-06 12:00:00', 0.01, 0.01*200, 200), # Win
        ]
        cursor.executemany("INSERT INTO trades (strategy, pair, is_open, open_date, close_date, close_profit, close_profit_abs, stake_amount) VALUES (?,?,?,?,?,?,?,?)",
                           trades_data_duoai + trades_data_other)
        conn.commit()
        conn.close()
        logger.info(f"Test database {db_path} created and populated.")


    async def run_test_strategy_manager():
        # Setup: Create dummy DB and mock objects
        setup_test_db(TEST_DB_PATH)

        # Instantiate StrategyManager with the test DB path
        # It will pick up FREQTRADE_DB_PATH from os.environ
        strategy_manager = StrategyManager()

        # Replace actual ParamsManager and BiasReflector with mocks for testing
        strategy_manager.params_manager = MockParamsManager()
        strategy_manager.bias_reflector = MockBiasReflector()

        test_strategy_id = "DUOAI_Strategy"
        other_strategy_id = "OtherStrategy"
        test_token = "ETH/USDT"

        print("\n--- Test StrategyManager with DB and Mocks ---")

        # 1. Test get_strategy_performance
        print(f"\nFetching performance for {test_strategy_id} from DB...")
        perf_duoai = strategy_manager.get_strategy_performance(test_strategy_id)
        print(f"Performance for {test_strategy_id}: {perf_duoai}")
        # DUOAI_Strategy: 2 wins, 1 loss out of 3 closed trades on ETH/USDT and BTC/USDT
        # Win rate = 2/3 = 0.666...
        # Avg profit = (0.02 - 0.01 + 0.03) / 3 = 0.04 / 3 = 0.01333...
        assert abs(perf_duoai['winRate'] - (2/3)) < 0.001
        assert abs(perf_duoai['avgProfit'] - (0.04/3)) < 0.001
        assert perf_duoai['tradeCount'] == 3

        print(f"\nFetching performance for {other_strategy_id} from DB...")
        perf_other = strategy_manager.get_strategy_performance(other_strategy_id)
        print(f"Performance for {other_strategy_id}: {perf_other}")
        # OtherStrategy: 2 wins out of 2 closed trades
        # Win rate = 2/2 = 1.0
        # Avg profit = (0.05 + 0.01) / 2 = 0.06 / 2 = 0.03
        assert abs(perf_other['winRate'] - 1.0) < 0.001
        assert abs(perf_other['avgProfit'] - 0.03) < 0.001
        assert perf_other['tradeCount'] == 2

        print(f"\nFetching performance for UnknownStrategy from DB...")
        perf_unknown = strategy_manager.get_strategy_performance("UnknownStrategy")
        print(f"Performance for UnknownStrategy: {perf_unknown}")
        assert perf_unknown['winRate'] == 0.0
        assert perf_unknown['avgProfit'] == 0.0
        assert perf_unknown['tradeCount'] == 0


        # 2. Test update_strategy_performance (should just log, not change DB data)
        print(f"\nAttempting to update performance for {test_strategy_id} (should only log)...")
        await strategy_manager.update_strategy_performance(test_strategy_id, {"winRate": 0.99, "avgProfit": 0.5, "tradeCount": 100})
        perf_after_update_attempt = strategy_manager.get_strategy_performance(test_strategy_id)
        assert abs(perf_after_update_attempt['winRate'] - (2/3)) < 0.001, "Performance should not change via update_strategy_performance"


        # 3. Test mutate_strategy
        print(f"\nMutating strategy {test_strategy_id} using MockParamsManager...")
        mock_proposal = {
            "strategyId": test_strategy_id,
            "adjustments": {
                "parameterChanges": {"emaPeriod": 25, "newParam": 123},
                "roi": {0: 0.12, 60: 0.08},
                "stoploss": -0.15,
                "trailingStop": {"value": 0.025, "offset": 0.005} # Updated structure
            },
            "confidence": 0.90,
            "rationale": "Test mutation via ParamsManager"
        }
        mutation_successful = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal)
        print(f"Mutation successful: {mutation_successful}")
        assert mutation_successful

        # Verify changes through MockParamsManager's internal state
        mutated_params_from_pm = await strategy_manager.params_manager.get_param("strategies", strategy_id=test_strategy_id)
        print(f"Parameters from MockParamsManager after mutation: {mutated_params_from_pm}")
        assert mutated_params_from_pm['buy']['emaPeriod'] == 25
        assert mutated_params_from_pm['buy']['newParam'] == 123
        assert mutated_params_from_pm['roi'] == {0: 0.12, 60: 0.08}
        assert mutated_params_from_pm['stoploss'] == -0.15
        # Assertions for the new trailing stop structure
        assert 'trailing' in mutated_params_from_pm, "Trailing params should exist"
        assert mutated_params_from_pm['trailing']['value'] == 0.025, "Trailing stop value incorrect"
        assert mutated_params_from_pm['trailing']['offset'] == 0.005, "Trailing stop offset incorrect"
        assert strategy_manager.params_manager.files_written.get(test_strategy_id) is True

        # Test case for missing trailingStop parts (optional, but good for robustness)
        print(f"\nMutating strategy {test_strategy_id} with partial trailingStop proposal...")
        partial_trailing_proposal = {
            "strategyId": test_strategy_id,
            "adjustments": {
                "trailingStop": {"value": 0.030} # Only value, offset should remain or be None
            },
            "confidence": 0.90,
            "rationale": "Test partial trailingStop mutation"
        }
        # Re-initialize a portion of mock params to simulate existing state for 'offset'
        # This depends on how MockParamsManager is structured. If it merges, this is fine.
        # If it overwrites the whole 'trailing' dict, then the previous 'offset' would be gone.
        # Based on current MockParamsManager, it updates specific keys, so previous offset should persist if not provided.
        # Let's ensure 'offset' was set by previous full mutation.
        # strategy_manager.params_manager.params[test_strategy_id]['trailing']['offset'] = 0.005 # Ensure it exists

        mutation_successful_partial = await strategy_manager.mutate_strategy(test_strategy_id, partial_trailing_proposal)
        assert mutation_successful_partial
        mutated_params_partial = await strategy_manager.params_manager.get_param("strategies", strategy_id=test_strategy_id)
        print(f"Parameters from MockParamsManager after partial trailingStop mutation: {mutated_params_partial}")
        assert mutated_params_partial['trailing']['value'] == 0.030 # New value
        assert mutated_params_partial['trailing']['offset'] == 0.005 # Offset from previous full mutation should persist


        # 4. Test get_best_strategy
        print(f"\nGetting best strategy for {test_token} (interval 5m)...")
        best_strat_info = await strategy_manager.get_best_strategy(test_token, "5m")
        print(f"Best strategy info: {best_strat_info}")
        assert best_strat_info is not None
        assert best_strat_info['id'] == test_strategy_id
        # Performance from DB
        assert abs(best_strat_info['performance']['winRate'] - (2/3)) < 0.001
        # Parameters from MockParamsManager (after mutation)
        assert best_strat_info['parameters']['buy']['emaPeriod'] == 25
        assert best_strat_info['parameters']['roi'] == {0: 0.12, 60: 0.08}
        # Bias from MockBiasReflector
        assert best_strat_info['bias'] == 0.75

        logger.info("All StrategyManager tests passed with DB and Mocks.")

        # Cleanup test DB
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        logger.info(f"Test database {TEST_DB_PATH} removed.")


    asyncio.run(run_test_strategy_manager())

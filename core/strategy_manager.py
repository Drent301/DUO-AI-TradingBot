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
# logger.setLevel(logging.INFO) # Logging level configured by application

from pathlib import Path # Import Path

# Define base directory for context if needed, e.g. project root
# Assuming this file (core/strategy_manager.py) is in: <project_root>/core/
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# FREQTRADE_DB_PATH will be used for the database connection
# Default location: user_data/freqtrade.sqlite relative to project root
USER_DATA_DIR = PROJECT_ROOT_DIR / "user_data"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure user_data directory exists
FREQTRADE_DB_PATH_DEFAULT = str(USER_DATA_DIR / 'freqtrade.sqlite')


class StrategyManager:
    """
    Manages strategy selection, mutation, and performance tracking using Freqtrade's database.
    """

    def __init__(self, db_path: Optional[str] = None):
        db_path_str = db_path or os.getenv('FREQTRADE_DB_PATH', FREQTRADE_DB_PATH_DEFAULT)
        self.db_path = Path(db_path_str)
        # Resolve to make it absolute. If db_path_str is already absolute, this does nothing.
        # If it's relative, it's resolved against the current working directory.
        # For consistency, if a relative path is given for db_path or via env var,
        # it might be better to resolve it against PROJECT_ROOT_DIR or USER_DATA_DIR.
        # However, Freqtrade itself might expect it relative to CWD or user_data.
        # For now, direct Path conversion and .resolve() is a good first step.
        # If FREQTRADE_DB_PATH_DEFAULT is used, it's already absolute or relative to USER_DATA_DIR.
        if not self.db_path.is_absolute():
            # If db_path is still relative (e.g. from input arg or env var),
            # assume it's relative to USER_DATA_DIR for robustness,
            # as this is a common Freqtrade convention.
            self.db_path = (USER_DATA_DIR / self.db_path).resolve()

        self.params_manager = ParamsManager()
        self.bias_reflector: Optional[BiasReflector] = None
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
            # sqlite3.connect can handle Path objects directly since Python 3.6
            conn = sqlite3.connect(self.db_path)
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
            performance['winRate'] = float(performance.get('winRate', 0.0) or 0.0)
            performance['avgProfit'] = float(performance.get('avgProfit', 0.0) or 0.0)
            performance['tradeCount'] = int(performance.get('tradeCount', 0) or 0)

            logger.info(f"Performance for {strategy_id}: {performance}")
            return performance

        except sqlite3.Error as e:
            logger.error(f"Database error while fetching performance for {strategy_id} from {self.db_path}: {e}", exc_info=True)
            return default_performance
        except Exception as e:
            logger.error(f"Unexpected error fetching performance for {strategy_id} from {self.db_path}: {e}", exc_info=True)
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
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Error connecting to Freqtrade API at {freqtrade_url} for {strategy_id}: {e}", exc_info=True)
        except aiohttp.ClientError as e:
            logger.error(f"AIOHTTP client error during Freqtrade API call for {strategy_id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during Freqtrade API call for {strategy_id}: {e}", exc_info=True)


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    # PROJECT_ROOT_DIR is defined at the top of the file
    dotenv_path = PROJECT_ROOT_DIR / '.env'
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
    else:
        logger.warning(f".env file not found at {dotenv_path} for __main__ test run.")


    # Setup basic logging for the test
    import sys # sys import for StreamHandler
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Define a test DB path and ensure it's used by StrategyManager
    TEST_DB_PATH = Path("test_freqtrade.sqlite").resolve() # Make it an absolute Path object
    os.environ['FREQTRADE_DB_PATH'] = str(TEST_DB_PATH) # Environment variables must be strings


    # --- Mock ParamsManager and BiasReflector for testing ---
    class MockParamsManager:
        def __init__(self, initial_params: Optional[Dict[str, Any]] = None, initial_strategy_ids: Optional[List[str]] = None):
            self.default_params_set = {
                "DUOAI_Strategy_V1": { # Renamed from DUOAI_Strategy to match other branch and test data
                    "id": "DUOAI_Strategy_V1",
                    "buy_params": {"emaPeriod": 20, "rsiThresholdBuy": 70}, # Changed from buy to buy_params
                    "sell_params": {"rsiThresholdSell": 70}, # Changed from sell to sell_params
                    "roi_table": {"0": 0.1, "30": 0.05, "60": 0.01}, # Changed from roi to roi_table and added 60m
                    "stoploss_value": -0.10, # Changed from stoploss to stoploss_value
                    "minTradeCountForScoring": 40,
                    "trailing_stop_positive": 0.01, # Added to match other branch
                    "trailing_stop_positive_offset": 0.02, # Added to match other branch
                    # Removed "trailing" dict as other branch uses flat trailing params in this mock
                },
                "DUOAI_Strategy_V2": { # Added to match other branch
                    "id": "DUOAI_Strategy_V2",
                    "buy_params": {"emaPeriod": 10, "rsiThresholdBuy": 65},
                    "sell_params": {"rsiThresholdSell": 75},
                    "roi_table": {"0": 0.15, "20": 0.08},
                    "stoploss_value": -0.05,
                    # minTradeCountForScoring will use default from get_best_strategy
                },
                "LowTrade_Strategy": { # Added to match other branch
                    "id": "LowTrade_Strategy",
                    "buy_params": {"emaPeriod": 5, "rsiThresholdBuy": 60},
                    "sell_params": {"rsiThresholdSell": 80},
                    "roi_table": {"0": 0.20},
                    "stoploss_value": -0.03,
                },
                # Keeping other strategy definitions from the 'main' branch for broader test coverage if needed later.
                "OtherStrategy": {
                    "id": "OtherStrategy",
                    "buy_params": {"emaPeriod": 50, "rsiThresholdBuy": 60},
                    "sell_params": {"rsiThresholdSell": 80},
                    "roi_table": {"0": 0.2, "30": 0.15, "60": 0.05}, "stoploss_value": -0.15,
                    # "trailing": {"enabled": False} # Removed as other branch mock uses flat structure for these
                },
                "DUOAI_Strategy_Zero": {
                    "id": "DUOAI_Strategy_Zero", "buy_params": {"param": "zero_trade_param"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}
                },
                "Strategy_A_Tie": {"id": "Strategy_A_Tie", "buy_params": {"param": "A"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}},
                "Strategy_B_Tie": {"id": "Strategy_B_Tie", "buy_params": {"param": "B"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}},
                "Strategy_C_Tie": {"id": "Strategy_C_Tie", "buy_params": {"param": "C"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}},
                "Strategy_D_Tie": {"id": "Strategy_D_Tie", "buy_params": {"param": "D"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}},
                "Strategy_E_Winner": {"id": "Strategy_E_Winner", "buy_params": {"param": "E"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}},
                "Strategy_F_Loser": {"id": "Strategy_F_Loser", "buy_params": {"param": "F"}, "stoploss_value": -0.10, "roi_table": {"0":0.1}}
            }
            # This will now be called strategies_config to match the get_param("strategies") call
            self.strategies_config = initial_params if initial_params is not None else self.default_params_set.copy()
            self.strategy_ids_override = initial_strategy_ids
            self.files_written = {}

        def update_params_data(self, new_params_data):
            self.strategies_config = new_params_data # Updated to strategies_config

        def override_strategy_ids(self, new_ids_list: Optional[List[str]]):
            self.strategy_ids_override = new_ids_list

        async def get_param(self, key: str, strategy_id: str = None): # Simplified signature
            if key == "strategies" and strategy_id is None:
                # This is called by get_best_strategy
                return self.strategies_config.copy()
            if key == "strategies" and strategy_id:
                # This might be called by mutate_strategy if it needs to get full original params first
                return self.strategies_config.get(strategy_id, {}).copy()
            # Fallback for direct parameter access if needed by other parts of tests
            if strategy_id:
                 strategy_conf = self.strategies_config.get(strategy_id, {})
                 # Attempt to get from buy_params, sell_params, or root
                 if key in strategy_conf.get("buy_params", {}):
                     return strategy_conf["buy_params"][key]
                 if key in strategy_conf.get("sell_params", {}):
                     return strategy_conf["sell_params"][key]
                 return strategy_conf.get(key)
            return None

        def get_all_strategy_ids(self) -> List[str]:
            if self.strategy_ids_override is not None:
                return self.strategy_ids_override
            return list(self.strategies_config.keys()) # Updated to strategies_config

        def get_strategy_params(self, strategy_id: str) -> Optional[Dict[str, Any]]:
            # This method is used by the 'main' branch version of get_best_strategy.
            # The merged get_best_strategy gets all strategies then iterates.
            # However, keeping it for compatibility or future test use.
            if strategy_id in self.strategies_config: # Updated to strategies_config
                return self.strategies_config[strategy_id].copy() # Updated to strategies_config
            return None

        async def set_param(self, key: str, value: Any, strategy_id: Optional[str] = None):
            if strategy_id:
                if strategy_id not in self.strategies_config: # Updated to strategies_config
                    self.strategies_config[strategy_id] = {"buy_params": {}, "sell_params": {}} # Updated to strategies_config

                # Adjusted logic to align with merged structure (buy_params, sell_params, etc.)
                # This mock will assume common keys go to 'buy_params' if not obviously 'roi_table' or 'stoploss_value' etc.
                if key == "emaPeriod" or "ThresholdBuy" in key : # Example, adapt as needed
                    if "buy_params" not in self.strategies_config[strategy_id]: self.strategies_config[strategy_id]["buy_params"] = {}
                    self.strategies_config[strategy_id]["buy_params"][key] = value
                elif "ThresholdSell" in key:
                    if "sell_params" not in self.strategies_config[strategy_id]: self.strategies_config[strategy_id]["sell_params"] = {}
                    self.strategies_config[strategy_id]["sell_params"][key] = value
                else: # roi_table, stoploss_value, or other direct params
                    self.strategies_config[strategy_id][key] = value
                logger.info(f"MockParamsManager: Set {key} to {value} for {strategy_id}")

        async def update_strategy_roi_sl_params(self, strategy_id: str,
                                                new_roi: Optional[Dict] = None,
                                                new_stoploss: Optional[float] = None,
                                                new_trailing_stop: Optional[float] = None,
                                                new_trailing_only_offset_is_reached: Optional[float] = None):
            if strategy_id not in self.strategies_config: # Updated to strategies_config
                self.strategies_config[strategy_id] = {} # Updated to strategies_config

            if new_roi is not None:
                self.strategies_config[strategy_id]['roi_table'] = new_roi
            if new_stoploss is not None:
                self.strategies_config[strategy_id]['stoploss_value'] = new_stoploss

            # Merged version uses flat trailing params
            if new_trailing_stop is not None:
                self.strategies_config[strategy_id]['trailing_stop_positive'] = new_trailing_stop
            if new_trailing_only_offset_is_reached is not None:
                self.strategies_config[strategy_id]['trailing_stop_positive_offset'] = new_trailing_only_offset_is_reached

            if new_roi or new_stoploss or new_trailing_stop or new_trailing_only_offset_is_reached:
                self.files_written[strategy_id] = True
                logger.info(f"MockParamsManager: Updated ROI/SL/Trailing for {strategy_id}")


    class MockBiasReflector:
        def get_bias_score(self, token: str, strategy_id: str) -> float:
            # Return a predictable bias score for testing
            if token == "ETH/USDT" and strategy_id == "DUOAI_Strategy": # Original name
                return 0.75
            if token == "ETH/USDT" and strategy_id == "DUOAI_Strategy_V1": # Name used in test data
                return 0.6 # Example bias for V1
            if token == "ETH/USDT" and strategy_id == "DUOAI_Strategy_V2":
                return 0.7 # Example bias for V2
            if token == "ETH/USDT" and strategy_id == "LowTrade_Strategy":
                 return 0.8 # Example bias for LowTrade
            return 0.5

    def setup_test_db(db_path: Path, custom_trades_data: Optional[List[tuple]] = None): # db_path is Path
        if db_path.exists():
            db_path.unlink() # Use unlink for Path objects
        conn = sqlite3.connect(db_path) # sqlite3.connect handles Path
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

        default_trades = trades_data_duoai + trades_data_other
        trades_to_insert = custom_trades_data if custom_trades_data is not None else default_trades

        if trades_to_insert: # Only insert if there's data
            cursor.executemany("INSERT INTO trades (strategy, pair, is_open, open_date, close_date, close_profit, close_profit_abs, stake_amount) VALUES (?,?,?,?,?,?,?,?)",
                               trades_to_insert)
        conn.commit()
        conn.close()
        # Using print for main block, logger might not be configured when this is called standalone in some contexts
        print(f"Test database {db_path} created/recreated and populated with {len(trades_to_insert)} trades.")


async def run_test_strategy_manager():
    # Setup: Create dummy DB and mock objects
    # Corrected path for .env when running this script directly
    # No need to load .env here if StrategyManager and ParamsManager don't directly use it in test setup
    # dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    # dotenv.load_dotenv(dotenv_path)

    # Setup basic logging for the test
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Define a test DB path and ensure it's used by StrategyManager
    TEST_DB_PATH = "test_freqtrade.sqlite"
    os.environ['FREQTRADE_DB_PATH'] = TEST_DB_PATH # StrategyManager will pick this up

    # Setup test DB with data for V1, V2, LowTrade strategies
    # Data for DUOAI_Strategy_V1 (50 trades, 60% WR, 0.01 avg profit)
    trades_v1 = [('DUOAI_Strategy_V1', 'ETH/USDT', 0, '2023-01-01', '2023-01-01', 0.02, 2, 100)] * 30
    trades_v1 += [('DUOAI_Strategy_V1', 'ETH/USDT', 0, '2023-01-02', '2023-01-02', -0.005, -0.5, 100)] * 20
    # Data for DUOAI_Strategy_V2 (60 trades, 70% WR, 0.015 avg profit)
    trades_v2 = [('DUOAI_Strategy_V2', 'ETH/USDT', 0, '2023-01-03', '2023-01-03', 0.025, 2.5, 100)] * 42
    trades_v2 += [('DUOAI_Strategy_V2', 'ETH/USDT', 0, '2023-01-04', '2023-01-04', -1.5/180, -1.5/1.8, 100)] * 18
    # Data for LowTrade_Strategy (10 trades, 80% WR, 0.02 avg profit)
    trades_low = [('LowTrade_Strategy', 'ETH/USDT', 0, '2023-01-05', '2023-01-05', 0.03, 3, 100)] * 8
    trades_low += [('LowTrade_Strategy', 'ETH/USDT', 0, '2023-01-06', '2023-01-06', -0.02, -2, 100)] * 2

    # Adding data for 'OtherStrategy' for broader compatibility with old tests if they were to be partially reused.
    # These trades won't affect the primary tests for V1, V2, LowTrade unless explicitly queried.
    trades_other = [
        ('OtherStrategy', 'LTC/USDT', 0, '2023-01-05 10:00:00', '2023-01-05 12:00:00', 0.05, 0.05*200, 200),
        ('OtherStrategy', 'LTC/USDT', 0, '2023-01-06 10:00:00', '2023-01-06 12:00:00', 0.01, 0.01*200, 200),
    ]
    all_trades_for_setup = trades_v1 + trades_v2 + trades_low + trades_other
    setup_test_db(TEST_DB_PATH, custom_trades_data=all_trades_for_setup) # Use the combined list

    strategy_manager = StrategyManager() # Will use TEST_DB_PATH via env var
    strategy_manager.params_manager = MockParamsManager() # Uses merged MockParamsManager
    strategy_manager.bias_reflector = MockBiasReflector() # Uses merged MockBiasReflector

    test_token = "ETH/USDT"
    test_interval = "5m"

    logger.info("--- Test StrategyManager (Merged get_best_strategy & Mocks) ---")

    # Verify DB setup for the main test strategies
    perf_v1 = strategy_manager.get_strategy_performance("DUOAI_Strategy_V1")
    assert perf_v1['tradeCount'] == 50, f"V1 TC: {perf_v1['tradeCount']}"
    assert abs(perf_v1['winRate'] - 0.60) < 0.001, f"V1 WR: {perf_v1['winRate']}"
    assert abs(perf_v1['avgProfit'] - 0.01) < 0.001, f"V1 AP: {perf_v1['avgProfit']}"

    perf_v2 = strategy_manager.get_strategy_performance("DUOAI_Strategy_V2")
    assert perf_v2['tradeCount'] == 60, f"V2 TC: {perf_v2['tradeCount']}"
    assert abs(perf_v2['winRate'] - 0.70) < 0.001, f"V2 WR: {perf_v2['winRate']}"
    assert abs(perf_v2['avgProfit'] - 0.015) < 0.001, f"V2 AP: {perf_v2['avgProfit']}"

    perf_low = strategy_manager.get_strategy_performance("LowTrade_Strategy")
    assert perf_low['tradeCount'] == 10, f"Low TC: {perf_low['tradeCount']}"
    assert abs(perf_low['winRate'] - 0.80) < 0.001, f"Low WR: {perf_low['winRate']}"
    assert abs(perf_low['avgProfit'] - 0.02) < 0.001, f"Low AP: {perf_low['avgProfit']}"

    # Test Case 1: Basic Selection (V2 should win based on scoring)
    # V1 Score: (0.6*0.7) + (0.01*10.0) + (0.6*0.1) = 0.42 + 0.1 + 0.06 = 0.58
    # V2 Score: (0.7*0.7) + (0.015*10.0) + (0.7*0.1) = 0.49 + 0.15 + 0.07 = 0.71
    # LowTrade Score (raw): (0.8*0.7) + (0.02*10.0) + (0.8*0.1) = 0.56 + 0.2 + 0.08 = 0.84. Penalized (10 < 50): 0.84 * 0.5 = 0.42
    logger.info("--- Test Case: Basic Selection (V2 expected) ---")
    best_strat_info = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_info is not None, "Should find a best strategy"
    assert best_strat_info['id'] == "DUOAI_Strategy_V2", f"Expected V2, got {best_strat_info['id']}. Score: {best_strat_info.get('calculated_score')}"
    assert abs(best_strat_info['calculated_score'] - 0.71) < 0.001, f"V2 Score mismatch: {best_strat_info['calculated_score']}"
    logger.info(f"Test Case Basic Selection Passed. Best: {best_strat_info['id']}")

    # Test Case 2: Low Trade Count Penalty
    # We expect LowTrade_Strategy to be penalized. V1 should win over it if V2 is made less attractive.
    logger.info("--- Test Case: Low Trade Count Penalty (V1 expected over LowTrade if V2 is weakened) ---")
    original_v2_config = strategy_manager.params_manager.strategies_config["DUOAI_Strategy_V2"]

    # Temporarily modify V2's performance to be worse than V1 for this test section
    original_get_performance = strategy_manager.get_strategy_performance
    def mock_get_performance_penalty_test(strategy_id_arg):
        if strategy_id_arg == "DUOAI_Strategy_V2": return {"winRate": 0.1, "avgProfit": 0.001, "tradeCount": 60} # Bad perf for V2
        return original_get_performance(strategy_id_arg)
    strategy_manager.get_strategy_performance = mock_get_performance_penalty_test

    # Recalculate scores:
    # V1 Score = 0.58 (as before)
    # V2 new score (bad perf): (0.1*0.7) + (0.001*10.0) + (0.7*0.1) = 0.07 + 0.01 + 0.07 = 0.15
    # LowTrade Penalized Score = 0.42 (as before)
    # Expected order: V1 (0.58) > LowTrade (0.42) > V2 (0.15)
    best_strat_penalty_test = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_penalty_test is not None
    assert best_strat_penalty_test['id'] == "DUOAI_Strategy_V1", f"Expected V1 due to V2 mod and LowTrade penalty, got {best_strat_penalty_test['id']}"

    strategy_manager.get_strategy_performance = original_get_performance # Restore
    # No need to restore V2 config in params_manager as we mocked get_strategy_performance
    logger.info(f"Test Case Low Trade Count Penalty Passed. Best: {best_strat_penalty_test['id']}")

    # Test Case 3: Strategy-Specific minTradeCountForScoring
    # V1 minTradeCountForScoring = 40. Actual trades = 50. Not penalized. Score = 0.58
    # V2 default threshold = 50. Give it 45 trades. Should be penalized.
    # Raw V2 score = 0.71. Penalized V2 score = 0.71 * 0.5 = 0.355
    # Expected order: V1 (0.58) > Penalized V2 (0.355) > LowTrade (0.42 without V2's data mock) or V1 > LowTrade > Penalized V2
    # Let's ensure LowTrade is also present with its original performance for this test.
    logger.info("--- Test Case: Strategy-Specific minTradeCountForScoring ---")

    original_get_perf_specific = strategy_manager.get_strategy_performance
    def mock_get_performance_v2_specific_thresh(strategy_id_arg):
        if strategy_id_arg == "DUOAI_Strategy_V2":
            return {"winRate": 0.7, "avgProfit": 0.015, "tradeCount": 45} # V2 has 45 trades
        # DUOAI_Strategy_V1 and LowTrade_Strategy use their actual performance from DB
        return original_get_perf_specific(strategy_id_arg)
    strategy_manager.get_strategy_performance = mock_get_performance_v2_specific_thresh

    # Scores:
    # V1 = 0.58 (50 trades vs threshold 40 from its config -> no penalty)
    # V2 (penalized) = ((0.7*0.7) + (0.015*10.0) + (0.7*0.1)) * 0.5 = (0.49 + 0.15 + 0.07) * 0.5 = 0.71 * 0.5 = 0.355
    # LowTrade (penalized) = 0.42 (10 trades vs default threshold 50 -> penalty)
    # Expected: V1 (0.58)
    best_strat_specific_thresh = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_specific_thresh is not None
    assert best_strat_specific_thresh['id'] == "DUOAI_Strategy_V1", f"Expected V1 due to V2 specific penalty, got {best_strat_specific_thresh['id']}"
    strategy_manager.get_strategy_performance = original_get_perf_specific # Restore
    logger.info(f"Test Case Strategy-Specific Threshold Passed. Best: {best_strat_specific_thresh['id']}")

    # Test Case 4: No Strategies Configured
    logger.info("--- Test Case: No Strategies Configured ---")
    original_strategies_config = strategy_manager.params_manager.strategies_config
    strategy_manager.params_manager.update_params_data({}) # Empty config
    best_strat_no_config = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_no_config is None, "Expected None when no strategies are configured"
    strategy_manager.params_manager.update_params_data(original_strategies_config) # Restore
    logger.info("Test Case No Strategies Configured Passed.")

    # Test Case 5: BiasReflector Unavailable
    logger.info("--- Test Case: BiasReflector Unavailable ---")
    original_bias_reflector = strategy_manager.bias_reflector
    strategy_manager.bias_reflector = None # Disable bias reflector
    # Scores with default bias 0.5 for all:
    # V1: (0.6*0.7) + (0.01*10.0) + (0.5*0.1) = 0.42 + 0.1 + 0.05 = 0.57
    # V2: (0.7*0.7) + (0.015*10.0) + (0.5*0.1) = 0.49 + 0.15 + 0.05 = 0.69
    # LowTrade (penalized): ((0.8*0.7) + (0.02*10.0) + (0.5*0.1)) * 0.5 = (0.56 + 0.2 + 0.05) * 0.5 = 0.81 * 0.5 = 0.405
    # Expected: V2 (0.69)
    best_strat_no_bias_reflector = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_no_bias_reflector is not None
    assert best_strat_no_bias_reflector['id'] == "DUOAI_Strategy_V2", f"Expected V2 with default bias, got {best_strat_no_bias_reflector['id']}"
    assert abs(best_strat_no_bias_reflector['calculated_score'] - 0.69) < 0.001
    strategy_manager.bias_reflector = original_bias_reflector # Restore
    logger.info("Test Case BiasReflector Unavailable Passed.")

    # Test mutate_strategy (with merged MockParamsManager)
    logger.info("--- Test Mutate Strategy (with merged MockParamsManager) ---")
    test_mutate_id = "DUOAI_Strategy_V1"
    mock_proposal_mutate = {
        "strategyId": test_mutate_id,
        "adjustments": {
            "parameterChanges": {"emaPeriod": 30}, # Should go into 'buy_params'
            "roi": {"0": 0.22}, # Should become 'roi_table'
            "stoploss": -0.22, # Should become 'stoploss_value'
            "trailingStop": {"value": 0.033, "offset": 0.011} # Flat trailing params
        }
    }
    await strategy_manager.mutate_strategy(test_mutate_id, mock_proposal_mutate)
    # Fetch all params for the strategy to check
    mutated_params_v1_all = await strategy_manager.params_manager.get_param("strategies", strategy_id=test_mutate_id)

    assert mutated_params_v1_all["buy_params"]["emaPeriod"] == 30
    assert mutated_params_v1_all["roi_table"] == {"0": 0.22}
    assert mutated_params_v1_all["stoploss_value"] == -0.22
    assert mutated_params_v1_all["trailing_stop_positive"] == 0.033
    assert mutated_params_v1_all["trailing_stop_positive_offset"] == 0.011
    logger.info("Test Mutate Strategy with merged mocks passed.")

    # Test zero trade count strategies (from 'main' branch tests)
    logger.info("--- Test Scenario: All Strategies Have Zero Trades (adapted) ---")
    zero_trade_strategy_name = "DUOAI_Strategy_Zero"
    # Setup DB with only open trades for relevant strategies
    setup_test_db(TEST_DB_PATH, custom_trades_data=[
        (zero_trade_strategy_name, 'ETH/USDT', 1, '2023-01-01', None, None, None, 100),
        ("DUOAI_Strategy_V1", 'ETH/USDT', 1, '2023-01-01', None, None, None, 100) # Also zero closed trades
    ])

    original_pm_config_zero = strategy_manager.params_manager.strategies_config
    ids_for_zero_test = [zero_trade_strategy_name, "DUOAI_Strategy_V1"]
    zero_trade_test_config = {
        k: original_pm_config_zero[k] for k in ids_for_zero_test if k in original_pm_config_zero
    }
    strategy_manager.params_manager.update_params_data(zero_trade_test_config)
    # get_best_strategy uses get_param("strategies") which gets keys from the config directly,
    # so override_strategy_ids is not strictly necessary if update_params_data sets the exact config.

    best_strat_zero_trades = await strategy_manager.get_best_strategy(test_token, test_interval)
    assert best_strat_zero_trades is not None, "Should select a strategy even if all have zero trades"
    # With scoring, both will have score 0 (plus bias). Bias for DUOAI_Strategy_Zero might be default 0.5.
    # Bias for DUOAI_Strategy_V1 for ETH/USDT is 0.6. So V1 might be picked.
    # Let's check the expected scores:
    # ZeroTrade: (0*0.7) + (0*10) + (0.5*0.1) = 0.05. Penalty factor 0.5 -> 0.025
    # V1: (0*0.7) + (0*10) + (0.6*0.1) = 0.06. Penalty factor 0.5 -> 0.03
    # So, DUOAI_Strategy_V1 is expected.
    assert best_strat_zero_trades['id'] == "DUOAI_Strategy_V1", f"Expected DUOAI_Strategy_V1, got {best_strat_zero_trades['id']}"
    assert best_strat_zero_trades['performance']['tradeCount'] == 0
    logger.info(f"Test Passed: All zero trades, selected {best_strat_zero_trades['id']}.")

    strategy_manager.params_manager.update_params_data(original_pm_config_zero) # Restore
    setup_test_db(TEST_DB_PATH, custom_trades_data=all_trades_for_setup) # Restore DB to full data


    logger.info("All StrategyManager tests (merged) passed.")

    # Cleanup test DB
    if TEST_DB_PATH.exists(): # TEST_DB_PATH is Path
        TEST_DB_PATH.unlink()
    logger.info(f"Test database {TEST_DB_PATH} removed.")


    asyncio.run(run_test_strategy_manager())

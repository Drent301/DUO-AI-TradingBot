# core/trade_logger.py
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the default path for Freqtrade's database
# Allow override via environment variable, e.g., for testing
FREQTRADE_DB_PATH_DEFAULT = 'freqtrade.sqlite'
FREQTRADE_DB_PATH = os.getenv('FREQTRADE_DB_PATH', FREQTRADE_DB_PATH_DEFAULT)

class TradeLogger:
    """
    Handles trade logging.
    Primary trade logging is delegated to Freqtrade's internal database.
    This class serves as a minimal placeholder for potential future advanced
    export functionalities or specific debugging needs.
    """

    def __init__(self, db_path: str = None):
        """
        Initializes the TradeLogger.
        The actual log file for individual trades is now less critical as Freqtrade DB is primary.
        """
        self.db_path = db_path or FREQTRADE_DB_PATH
        logger.info(f"TradeLogger initialized. Using DB path: {self.db_path}. "
                    "Primary trade logging is handled by Freqtrade's database.")

    def log_trade(self, trade_data: dict):
        """
        Placeholder for logging a trade.
        Actual trade data is logged by Freqtrade into its database.
        This method can be expanded for additional debugging or export if needed.
        """
        logger.info(f"Trade action occurred (data: {trade_data.get('pair', 'N/A')}). "
                    "This is a placeholder log; Freqtrade DB is the primary record.")
        # Secondary logging to a JSON file (optional, can be removed or kept for debugging)
        # self._log_to_json(trade_data) # Commented out to simplify

    def _log_to_json(self, trade_data: dict):
        """
        (Optional) Logs trade data to a JSON file for auxiliary debugging.
        """
        # This functionality is secondary and can be customized or removed.
        # For example, log to a file in a 'logs' directory:
        # log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        # os.makedirs(log_dir, exist_ok=True)
        # trade_log_file = os.path.join(log_dir, "aux_trade_log.json")
        # try:
        #     with open(trade_log_file, 'a', encoding='utf-8') as f:
        #         json.dump({datetime.now().isoformat(): trade_data}, f, indent=2)
        #         f.write('\n')
        # except IOError as e:
        #     logger.error(f"Error writing to auxiliary JSON trade log: {e}")
        pass

    async def get_all_trades_from_db(self) -> list[dict]:
        """
        Fetches all trade records from the Freqtrade database.
        Returns a list of dictionaries, where each dictionary represents a trade.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Using asyncio.to_thread to run the blocking DB call in a separate thread
            df = await asyncio.to_thread(pd.read_sql_query, "SELECT * FROM trades", conn)
            return df.to_dict(orient='records')
        except sqlite3.Error as e:
            logger.error(f"Database error while fetching all trades: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching all trades: {e}")
            return []
        finally:
            if conn:
                # Ensure connection is closed in the same thread it was created
                await asyncio.to_thread(conn.close)


# --- Test Suite ---
TEST_DB_PATH = "test_trades_logger.sqlite"

def setup_test_db(db_path: str):
    """Creates and populates a dummy Freqtrade database for testing."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create a 'trades' table (simplified version of Freqtrade's schema for this test)
    # A more accurate test would use the exact Freqtrade schema.
    cursor.execute("""
    CREATE TABLE trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT NOT NULL,
        amount REAL,
        open_rate REAL,
        close_rate REAL,
        open_date DATETIME,
        close_date DATETIME,
        stake_amount REAL,
        profit_ratio REAL,
        profit_abs REAL,
        sell_reason TEXT,
        strategy TEXT,
        is_open INTEGER DEFAULT 1
    )
    """)
    sample_trades = [
        ('ETH/USDT', 1.0, 2000.0, 2100.0, datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 1, 12, 0, 0), 2000.0, 0.05, 100.0, 'roi', 'SampleStrategy', 0),
        ('BTC/USDT', 0.1, 30000.0, 29000.0, datetime(2023, 1, 2, 10, 0, 0), datetime(2023, 1, 2, 12, 0, 0), 3000.0, -0.0333, -100.0, 'stop_loss', 'SampleStrategy', 0),
        ('LTC/USDT', 10.0, 70.0, None, datetime(2023, 1, 3, 10, 0, 0), None, 700.0, None, None, None, 'AnotherStrategy', 1)
    ]
    cursor.executemany("""
    INSERT INTO trades (pair, amount, open_rate, close_rate, open_date, close_date, stake_amount, profit_ratio, profit_abs, sell_reason, strategy, is_open)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_trades)
    conn.commit()
    conn.close()
    logger.info(f"Test database {db_path} created and populated with {len(sample_trades)} trades.")

async def run_test_trade_logger():
    """Tests the TradeLogger functionality."""
    # Set environment variable for DB path to use the test DB
    os.environ['FREQTRADE_DB_PATH'] = TEST_DB_PATH

    # Setup: Create dummy DB
    setup_test_db(TEST_DB_PATH)

    # Instantiate TradeLogger (it will pick up FREQTRADE_DB_PATH from os.environ)
    trade_logger = TradeLogger()

    print("\n--- Test TradeLogger ---")

    # 1. Test log_trade (placeholder behavior)
    print("\nTesting log_trade (placeholder)...")
    test_trade_data = {"pair": "XRP/USDT", "action": "buy", "price": 0.5, "amount": 1000}
    trade_logger.log_trade(test_trade_data)
    # Verification: Primarily ensures it runs without error. Log output can be checked manually or with advanced logging capture.
    print("log_trade called successfully.")

    # 2. Test get_all_trades_from_db
    print("\nTesting get_all_trades_from_db...")
    all_trades = await trade_logger.get_all_trades_from_db()
    print(f"Retrieved {len(all_trades)} trades from the database.")

    assert len(all_trades) == 3, f"Expected 3 trades, got {len(all_trades)}"

    # Verify some data from the first retrieved trade (order might not be guaranteed by SQL without ORDER BY)
    # For robust testing, sort or find specific trades if order is not fixed.
    # Here, we assume the order of insertion is preserved for simplicity.
    first_trade = next((t for t in all_trades if t['pair'] == 'ETH/USDT'), None)
    assert first_trade is not None, "ETH/USDT trade not found"
    assert first_trade['strategy'] == 'SampleStrategy'
    assert first_trade['profit_ratio'] == 0.05
    assert first_trade['is_open'] == 0

    opened_trade = next((t for t in all_trades if t['is_open'] == 1), None)
    assert opened_trade is not None, "Open trade not found"
    assert opened_trade['pair'] == 'LTC/USDT'

    print("get_all_trades_from_db successfully retrieved and validated data.")

    # Cleanup test DB
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    logger.info(f"Test database {TEST_DB_PATH} removed.")

    # Unset environment variable if necessary, or it just affects this run
    del os.environ['FREQTRADE_DB_PATH']

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    asyncio.run(run_test_trade_logger())

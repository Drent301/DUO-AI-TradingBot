"""
PatternPerformanceAnalyzer Module

Analyses the performance of detected trading patterns by correlating pattern entry logs
with closed trade results from a Freqtrade database.
It calculates metrics such as total trades, win rate, and average profit percentage
for each pattern that contributed to an entry decision.
"""
import json
import logging
import os # Keep for os.getenv if used, or other non-path operations
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path # Import Path
import pandas as pd
from typing import List, Dict, Any, Optional

# Assuming ParamsManager is in core.params_manager
from core.params_manager import ParamsManager # Actual import if running within a larger system

logger = logging.getLogger(__name__)
# Default log level, can be overridden by application config
# logger.setLevel(logging.INFO) # Commented out to allow external configuration

# Define base directory for context if needed, e.g. project root
# Assuming this file (core/pattern_performance_analyzer.py) is in: <project_root>/core/
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent

# Default paths if not provided or found in params_manager
DEFAULT_USER_DATA_DIR = PROJECT_ROOT_DIR / "user_data"
DEFAULT_FREQTRADE_DB_PATH = DEFAULT_USER_DATA_DIR / "freqtrade.sqlite"
DEFAULT_PATTERN_LOG_PATH = DEFAULT_USER_DATA_DIR / "logs" / "pattern_performance_log.json"
DEFAULT_MATCH_WINDOW_MINUTES = 2 # Default time window for matching logs to trades

class PatternPerformanceAnalyzer:
    """
    Analyzes the performance of trading patterns.

    This class loads logs of patterns that contributed to trade entries and
    correlates them with closed trade data from a Freqtrade database.
    It then calculates performance metrics for each pattern, such as
    win rate and average profit percentage.
    """
    def __init__(self, params_manager: Optional[Any] = None,
                 freqtrade_db_path: Optional[str] = None,
                 pattern_log_path: Optional[str] = None):
        """
        Initializes the PatternPerformanceAnalyzer.

        Args:
            params_manager: An instance of ParamsManager for fetching configuration.
                            If None, default paths and settings will be used.
            freqtrade_db_path: Path to the Freqtrade SQLite database.
                               Overrides value from params_manager or default if provided.
            pattern_log_path: Path to the JSONL file containing pattern entry logs.
                              Overrides value from params_manager or default if provided.
        """
        self.params_manager = params_manager

        # Determine paths and configuration, prioritizing constructor arguments,
        # then params_manager, then class defaults.
        if freqtrade_db_path:
            self.freqtrade_db_path = freqtrade_db_path
        elif self.params_manager:
            self.freqtrade_db_path = self.params_manager.get_param("freqtradeDbPathAnalyzer", default=DEFAULT_FREQTRADE_DB_PATH)
        else:
            self.freqtrade_db_path = DEFAULT_FREQTRADE_DB_PATH

        if pattern_log_path:
            self.pattern_log_path = pattern_log_path
        elif self.params_manager:
            self.pattern_log_path = self.params_manager.get_param("patternPerformanceLogPath", default=DEFAULT_PATTERN_LOG_PATH)
        else:
            self.pattern_log_path = DEFAULT_PATTERN_LOG_PATH

        if self.params_manager:
            self.match_window_minutes = self.params_manager.get_param("patternLogTradeMatchWindowMinutes", default=DEFAULT_MATCH_WINDOW_MINUTES)
        else:
            self.match_window_minutes = DEFAULT_MATCH_WINDOW_MINUTES

        # Adjust relative paths to be absolute, assuming 'user_data' implies a path relative to the project root or CWD.
        # This makes paths more robust.
        self.freqtrade_db_path = Path(self.freqtrade_db_path)
        if not self.freqtrade_db_path.is_absolute():
            self.freqtrade_db_path = (PROJECT_ROOT_DIR / self.freqtrade_db_path).resolve()

        self.pattern_log_path = Path(self.pattern_log_path)
        if not self.pattern_log_path.is_absolute():
            self.pattern_log_path = (PROJECT_ROOT_DIR / self.pattern_log_path).resolve()


        logger.info(f"PatternPerformanceAnalyzer initialized. DB: '{self.freqtrade_db_path}', Log: '{self.pattern_log_path}', Window: {self.match_window_minutes} mins.")

    def _load_pattern_entry_logs(self) -> List[Dict[str, Any]]:
        """
        Loads pattern entry logs from the specified JSONL file.
        """
        logs: List[Dict[str, Any]] = []
        if not self.pattern_log_path.exists(): # Use Path.exists()
            logger.warning(f"Pattern entry log file not found: {self.pattern_log_path}")
            return logs

        try:
            with self.pattern_log_path.open('r', encoding='utf-8') as f: # Use Path.open()
                for line_number, line in enumerate(f, 1):
                    try:
                        log_entry = json.loads(line.strip())
                        if 'entry_timestamp' in log_entry and isinstance(log_entry['entry_timestamp'], str):
                            dt_obj = datetime.fromisoformat(log_entry['entry_timestamp'])
                            if dt_obj.tzinfo is None:
                                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                            log_entry['entry_timestamp'] = dt_obj
                        logs.append(log_entry)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON from line {line_number} in {self.pattern_log_path}: {e}. Line: '{line.strip()[:100]}...'", exc_info=True)
                    except ValueError as e:
                         logger.error(f"Error converting timestamp from line {line_number} in {self.pattern_log_path}: {e}. Entry: {log_entry}", exc_info=True)

        except IOError as e:
            logger.error(f"Error reading pattern log file {self.pattern_log_path}: {e}", exc_info=True)

        logger.info(f"Loaded {len(logs)} pattern entry logs from {self.pattern_log_path}")
        return logs

    def _load_closed_trades(self) -> pd.DataFrame:
        """
        Loads closed trades from the Freqtrade SQLite database.
        """
        if not self.freqtrade_db_path.exists(): # Use Path.exists()
            logger.warning(f"Freqtrade database not found: {self.freqtrade_db_path}")
            return pd.DataFrame()

        try:
            # Connect in read-only mode to prevent accidental writes
            # sqlite3.connect with URI=True expects a string path.
            conn = sqlite3.connect(f"file:{str(self.freqtrade_db_path)}?mode=ro", uri=True)
            query = "SELECT id, pair, open_date, close_date, profit_ratio FROM trades WHERE is_open = 0"
            df = pd.read_sql_query(query, conn)
            conn.close()

            df['open_date'] = pd.to_datetime(df['open_date'], utc=True)
            df['close_date'] = pd.to_datetime(df['close_date'], utc=True)

            logger.info(f"Loaded {len(df)} closed trades from {self.freqtrade_db_path}")
            return df
        except sqlite3.Error as e:
            logger.error(f"SQLite error while loading closed trades from {self.freqtrade_db_path}: {e}", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error loading closed trades from {self.freqtrade_db_path}: {e}", exc_info=True)
            return pd.DataFrame()

    def analyze_pattern_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes pattern performance by matching entry logs with closed trades.

        Calculates metrics like total trades, win rate, and average profit percentage
        for each pattern identified in the entry logs.

        Returns:
            A dictionary where keys are pattern names and values are dictionaries
            of their performance metrics. Example:
            {
                "cnn_5m_bullFlag": {
                    "total_trades": 10,
                    "wins": 6,
                    "total_profit_pct": 0.05, # Sum of profit_ratios
                    "win_rate_pct": 60.0,
                    "avg_profit_pct": 0.5
                }, ...
            }
        """
        pattern_entry_logs = self._load_pattern_entry_logs()
        closed_trades_df = self._load_closed_trades()

        if not pattern_entry_logs or closed_trades_df.empty:
            logger.warning("No pattern logs or closed trades to analyze. Returning empty metrics.")
            return {}

        pattern_metrics: Dict[str, Dict[str, Any]] = {}
        time_window = timedelta(minutes=self.match_window_minutes)

        # To prevent matching a single trade to multiple log entries if they are very close.
        # This assumes one log entry should correspond to one trade.
        matched_trade_ids = set()

        for log_entry in pattern_entry_logs:
            log_pair = log_entry.get('pair')
            log_ts = log_entry.get('entry_timestamp') # Should be datetime object from _load_pattern_entry_logs
            contributing_patterns = log_entry.get('contributing_patterns', [])

            if not log_pair or not isinstance(log_ts, datetime) or not contributing_patterns:
                logger.debug(f"Skipping log entry due to missing critical data or malformed timestamp: {log_entry.get('entry_timestamp')}")
                continue

            # Ensure log_ts is timezone-aware (UTC) if it's naive, to match Freqtrade's DB dates.
            # This should ideally be handled definitively in _load_pattern_entry_logs.
            if log_ts.tzinfo is None:
                log_ts = log_ts.replace(tzinfo=timezone.utc)

            # Filter trades by pair for efficiency before iterating
            relevant_trades = closed_trades_df[closed_trades_df['pair'] == log_pair].copy() # Use .copy() to avoid SettingWithCopyWarning

            match_found_for_log = False
            # Sort relevant_trades by proximity to log_ts to prefer closest match if multiple exist within window
            # This is important if a simple first-match-wins is used.
            if not relevant_trades.empty:
                relevant_trades['time_diff'] = (relevant_trades['open_date'] - log_ts).abs()
                relevant_trades.sort_values(by='time_diff', inplace=True)

            for _, trade_row in relevant_trades.iterrows():
                trade_open_ts = trade_row['open_date'] # Already tz-aware UTC from _load_closed_trades
                trade_id = trade_row['id']

                if trade_id in matched_trade_ids: # Skip if this trade was already matched to a previous log
                    continue

                # Check if the trade's open_date is within the time_window of the log's entry_timestamp
                if abs(log_ts - trade_open_ts) <= time_window:
                    profit_pct = trade_row['profit_ratio'] # This is a ratio, e.g., 0.02 for 2%

                    # Attribute this trade's outcome to all contributing patterns in the log
                    for pattern_name in contributing_patterns:
                        if pattern_name not in pattern_metrics:
                            pattern_metrics[pattern_name] = {
                                'total_trades': 0,
                                'wins': 0,
                                'total_profit_pct': 0.0, # Sum of profit_ratios
                                'related_trade_ids': []
                            }

                        pattern_metrics[pattern_name]['total_trades'] += 1
                        if profit_pct > 0: # A trade is a "win" if profit_ratio > 0
                            pattern_metrics[pattern_name]['wins'] += 1
                        pattern_metrics[pattern_name]['total_profit_pct'] += profit_pct
                        pattern_metrics[pattern_name]['related_trade_ids'].append(int(trade_id)) # Store as int

                    matched_trade_ids.add(trade_id) # Mark this trade as matched
                    match_found_for_log = True
                    logger.debug(f"Matched log {log_pair} @ {log_ts} with trade ID {trade_id} (open: {trade_open_ts}), patterns: {contributing_patterns}, profit_ratio: {profit_pct:.4f}")
                    break # Move to the next log entry once a unique trade is matched

            if not match_found_for_log:
                logger.debug(f"No unique matching trade found for log entry: {log_pair} @ {log_ts} within +/- {self.match_window_minutes} minutes.")


        # Calculate final derived metrics (win_rate, avg_profit_pct)
        if not pattern_metrics:
            logger.info("No pattern metrics were generated (no pattern entry logs were successfully matched to trades).")
            return {}

        for pattern_name, metrics_data in pattern_metrics.items(): # Use metrics_data to avoid conflict
            if metrics_data['total_trades'] > 0:
                win_rate = (metrics_data['wins'] / metrics_data['total_trades']) * 100.0
                avg_profit_pct = (metrics_data['total_profit_pct'] / metrics_data['total_trades']) * 100.0 # Convert ratio to percentage
            else:
                win_rate = 0.0
                avg_profit_pct = 0.0

            pattern_metrics[pattern_name]['win_rate_pct'] = win_rate
            pattern_metrics[pattern_name]['avg_profit_pct'] = avg_profit_pct
            # Optionally, remove intermediate calculation fields if they are not needed in the final output
            # For example: del pattern_metrics[pattern_name]['related_trade_ids'] (if too verbose)

        logger.info("Pattern Performance Analysis Summary:")
        for pattern_name, metrics_data in pattern_metrics.items(): # Use metrics_data
            logger.info(
                f"Pattern: {pattern_name:<30} | "
                f"Trades: {metrics['total_trades']:<5} | "
                f"Win Rate: {metrics['win_rate_pct']:>6.2f}% | "
                f"Avg Profit: {metrics['avg_profit_pct']:>6.2f}%"
            )

        return pattern_metrics

if __name__ == '__main__':
    # Example Usage (requires setup of dummy files or ParamsManager)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define dummy paths using Path objects relative to where the script is run,
    # or use PROJECT_ROOT_DIR for more robustness if structure is known.
    # For this example, assume user_data is at project root.
    test_user_data_dir = PROJECT_ROOT_DIR / "user_data"
    dummy_log_path = test_user_data_dir / "logs" / "pattern_performance_log_test.json"
    dummy_log_path.parent.mkdir(parents=True, exist_ok=True) # Use Path.mkdir()

    dummy_log_entries = [
        {"pair": "ETH/USDT", "entry_timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), "contributing_patterns": ["cnn_5m_bullFlag", "rule_morningStar"], "decision_details": {}},
        {"pair": "ETH/USDT", "entry_timestamp": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(), "contributing_patterns": ["cnn_1h_bearishEngulfing"], "decision_details": {}},
        {"pair": "BTC/USDT", "entry_timestamp": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(), "contributing_patterns": ["rule_bullishEngulfing"], "decision_details": {}},
    ]
    with dummy_log_path.open('w', encoding='utf-8') as f: # Use Path.open()
        for entry in dummy_log_entries:
            json.dump(entry, f)
            f.write('\n')

    # Create dummy freqtrade.sqlite
    dummy_db_path = test_user_data_dir / "freqtrade_test.sqlite"
    if dummy_db_path.exists(): dummy_db_path.unlink() # Use Path.unlink()
    conn = sqlite3.connect(str(dummy_db_path)) # sqlite3.connect needs string
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, pair TEXT, is_open INTEGER,
            open_date TIMESTAMP, close_date TIMESTAMP, profit_ratio REAL
        )
    """)
    # Trade 1: Matches first log for ETH/USDT
    cursor.execute("INSERT INTO trades VALUES (?,?,?,?,?,?)",
                   (1, "ETH/USDT", 0,
                    (datetime.now(timezone.utc) - timedelta(hours=2, minutes=1)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(hours=1, minutes=30)).isoformat(), 0.02))
    # Trade 2: Matches second log for ETH/USDT (loss)
    cursor.execute("INSERT INTO trades VALUES (?,?,?,?,?,?)",
                   (2, "ETH/USDT", 0,
                    (datetime.now(timezone.utc) - timedelta(hours=1, minutes=1)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(), -0.01))
    # Trade 3: Matches BTC/USDT log
    cursor.execute("INSERT INTO trades VALUES (?,?,?,?,?,?)",
                   (3, "BTC/USDT", 0,
                    (datetime.now(timezone.utc) - timedelta(minutes=30, seconds=30)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), 0.05))
    # Trade 4: Unmatched trade
    cursor.execute("INSERT INTO trades VALUES (?,?,?,?,?,?)",
                   (4, "ADA/USDT", 0,
                    (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
                    (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(), 0.03))
    conn.commit()
    conn.close()

    # Create analyzer instance (without ParamsManager for this direct test)
    # Pass paths as strings, as constructor expects Optional[str]
    analyzer = PatternPerformanceAnalyzer(
        freqtrade_db_path=str(dummy_db_path),
        pattern_log_path=str(dummy_log_path)
    )

    # Set a specific match window for testing
    analyzer.match_window_minutes = 5 # Test with a 5-minute window

    metrics = analyzer.analyze_pattern_performance()

    # Expected output based on dummy data and 5 min window:
    # Log 1 (ETH/USDT, cnn_5m_bullFlag, rule_morningStar) matches Trade 1 (profit 0.02)
    #   - cnn_5m_bullFlag: trades=1, wins=1, total_profit=0.02
    #   - rule_morningStar: trades=1, wins=1, total_profit=0.02
    # Log 2 (ETH/USDT, cnn_1h_bearishEngulfing) matches Trade 2 (profit -0.01)
    #   - cnn_1h_bearishEngulfing: trades=1, wins=0, total_profit=-0.01
    # Log 3 (BTC/USDT, rule_bullishEngulfing) matches Trade 3 (profit 0.05)
    #   - rule_bullishEngulfing: trades=1, wins=1, total_profit=0.05

    # Expected metrics:
    # cnn_5m_bullFlag: trades=1, wins=1, tot_profit=0.02 -> win_rate=100%, avg_profit=2.0%
    # rule_morningStar: trades=1, wins=1, tot_profit=0.02 -> win_rate=100%, avg_profit=2.0%
    # cnn_1h_bearishEngulfing: trades=1, wins=0, tot_profit=-0.01 -> win_rate=0%, avg_profit=-1.0%
    # rule_bullishEngulfing: trades=1, wins=1, tot_profit=0.05 -> win_rate=100%, avg_profit=5.0%

    logger.info("--- Verification of Dummy Run ---")
    expected_results = {
        "cnn_5m_bullFlag": {"total_trades": 1, "win_rate_pct": 100.0, "avg_profit_pct": 2.0},
        "rule_morningStar": {"total_trades": 1, "win_rate_pct": 100.0, "avg_profit_pct": 2.0},
        "cnn_1h_bearishEngulfing": {"total_trades": 1, "win_rate_pct": 0.0, "avg_profit_pct": -1.0},
        "rule_bullishEngulfing": {"total_trades": 1, "win_rate_pct": 100.0, "avg_profit_pct": 5.0},
    }

    for pattern, expected_metric_values in expected_results.items():
        assert pattern in metrics, f"Pattern {pattern} missing from results."
        for metric_key, expected_val in expected_metric_values.items():
            actual_val = metrics[pattern].get(metric_key)
            assert actual_val is not None, f"Metric {metric_key} missing for pattern {pattern}."
            assert abs(actual_val - expected_val) < 0.001, \
                f"Metric mismatch for {pattern}/{metric_key}. Expected {expected_val}, Got {actual_val}"
        logger.info(f"Verified: {pattern} matches expected metrics.")

    logger.info("Dummy run verification successful.")

    # Cleanup dummy files
    # os.remove(dummy_log_path) # Keep for inspection if needed
    # os.remove(dummy_db_path)
    logger.info(f"Kept dummy files for inspection: {dummy_log_path}, {dummy_db_path}")

```

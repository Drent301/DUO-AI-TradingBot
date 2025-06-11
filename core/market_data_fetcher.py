# core/market_data_fetcher.py
import json
import logging
import os
from typing import Dict, Any, Optional, List
import requests
import csv
import time
from datetime import datetime, timezone


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.json')
# DATA_DIR is now configured via config file or defaults to project_root/data/binance


class BinanceDataFetcher:
    """
    Fetches historical market data from Binance and can save it to CSV,
    including logic for data resumption.
    """
    KLINE_CSV_FIELDNAMES = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]

    def __init__(self):
        config_data = self._load_config()
        if config_data is None:
            # _load_config already logs errors, just need to handle the None case
            raise ValueError("Failed to load configuration. BinanceDataFetcher cannot be initialized.")

        self.api_url = config_data.get("data_sources", {}).get("binance", {}).get("api_url")
        if not self.api_url:
            logger.error("Binance API URL not found in config. Cannot initialize BinanceDataFetcher.")
            raise ValueError("Binance API URL not configured")

        raw_data_dir = config_data.get("data_sources", {}).get("binance", {}).get("data_directory")
        if raw_data_dir:
            self.data_dir = os.path.abspath(raw_data_dir)
        else:
            project_config_dir = os.path.dirname(CONFIG_FILE_PATH)
            project_root = os.path.dirname(project_config_dir)
            self.data_dir = os.path.abspath(os.path.join(project_root, "data", "binance"))

        logger.info(f"BinanceDataFetcher initialized with API URL: {self.api_url}")
        logger.info(f"BinanceDataFetcher: Using data directory: {self.data_dir}")

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Loads the main configuration file."""
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {CONFIG_FILE_PATH}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {CONFIG_FILE_PATH}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading configuration: {e}")
        return None

    def _interval_to_milliseconds(self, interval: str) -> Optional[int]:
        multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
        try:
            unit = interval[-1]
            value = int(interval[:-1])
            if unit in multipliers:
                return value * multipliers[unit] * 1000
            else:
                logger.error(f"Unsupported interval unit: {unit} in interval {interval}")
                return None
        except (ValueError, TypeError) as specific_err:
            logger.error(f"Error converting interval '{interval}' to milliseconds: {specific_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error converting interval '{interval}' to milliseconds: {e}")
            return None

    def _get_latest_saved_opentime(self, symbol: str, interval: str) -> Optional[int]:
        directory_path = os.path.join(self.data_dir, symbol.upper(), interval)
        if not os.path.exists(directory_path):
            logger.info(f"No data directory found for {symbol}/{interval} at {directory_path}. No prior data.")
            return None

        latest_file_start_time = -1
        target_file = None

        try:
            for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):
                    parts = filename.replace(".csv", "").split('_')
                    if len(parts) >= 3:
                        try:
                            file_start_time = int(parts[-1])
                            if file_start_time > latest_file_start_time:
                                latest_file_start_time = file_start_time
                                target_file = os.path.join(directory_path, filename)
                        except ValueError:
                            logger.warning(f"Could not parse timestamp from filename: {filename}")

            if not target_file:
                logger.info(f"No CSV files found or parsable in {directory_path} for {symbol}/{interval}.")
                return None

            logger.info(f"Identified latest batch file by name: {target_file}")

            max_open_time_in_file = -1
            with open(target_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        open_time = int(row['open_time'])
                        if open_time > max_open_time_in_file:
                            max_open_time_in_file = open_time
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping row in {target_file} due to parsing error: {e} - Row: {row}")

            return max_open_time_in_file if max_open_time_in_file != -1 else None

        except (OSError, IOError) as file_err:
            logger.error(f"File system error getting latest saved open time for {symbol}/{interval}: {file_err}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting latest saved open time for {symbol}/{interval}: {e}")
            return None

    def get_klines(self, symbol: str, interval: str, start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None, limit: int = 500) -> List[Dict[str, Any]]:
        endpoint = f"{self.api_url}/klines"
        params = {
            "symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)
        }
        if start_time_ms: params["startTime"] = start_time_ms
        if end_time_ms: params["endTime"] = end_time_ms

        logger.debug(f"Fetching klines from {endpoint} with params: {params}")
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            formatted_data = []
            for kline_item in data:
                formatted_data.append({
                    "open_time": int(kline_item[0]), "open": float(kline_item[1]),
                    "high": float(kline_item[2]), "low": float(kline_item[3]),
                    "close": float(kline_item[4]), "volume": float(kline_item[5]),
                    "close_time": int(kline_item[6]), "quote_asset_volume": float(kline_item[7]),
                    "number_of_trades": int(kline_item[8]),
                    "taker_buy_base_asset_volume": float(kline_item[9]),
                    "taker_buy_quote_asset_volume": float(kline_item[10])
                })
            return formatted_data
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error: {http_err} - Response: {http_err.response.text if http_err.response else 'N/A'} for {symbol} with params {params}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error for {symbol}: {req_err}")
        except ValueError as json_err:
            logger.error(f"JSON decode error for {symbol}: {json_err}")
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
        return []

    def save_klines_to_csv(self, klines: List[Dict[str, Any]], symbol: str, interval: str) -> Optional[str]:
        if not klines:
            logger.warning(f"No k-line data for {symbol}/{interval} to save.")
            return None
        try:
            directory_path = os.path.join(self.data_dir, symbol.upper(), interval)
            os.makedirs(directory_path, exist_ok=True)
            first_kline_open_time = klines[0]['open_time']
            filename = f"{symbol.upper()}_{interval}_{first_kline_open_time}.csv"
            filepath = os.path.join(directory_path, filename)

            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.KLINE_CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerows(klines)
            logger.info(f"Saved {len(klines)} klines to {filepath}")
            return filepath
        except IOError as io_err:
            logger.error(f"IOError saving CSV for {symbol}/{interval}: {io_err}")
        except Exception as e:
            logger.error(f"Error saving CSV for {symbol}/{interval}: {e}")
        return None

    def fetch_and_save_historical_data(self, symbol: str, interval: str, start_date_str: str, end_date_str: Optional[str] = None):
        logger.info(f"Starting historical data fetch for {symbol}/{interval} from {start_date_str} to {end_date_str or 'now'}.")
        date_format = "%Y-%m-%d"; datetime_format = "%Y-%m-%d %H:%M:%S"
        try:
            start_dt_obj = datetime.strptime(start_date_str, datetime_format if ' ' in start_date_str else date_format)
            user_start_ts_ms = int(start_dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError as e:
            logger.error(f"Invalid start_date_str: {start_date_str}. Error: {e}"); return

        target_end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if end_date_str:
            try:
                end_dt_obj = datetime.strptime(end_date_str, datetime_format if ' ' in end_date_str else date_format)
                target_end_ts_ms = int(end_dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError as e:
                logger.error(f"Invalid end_date_str: {end_date_str}. Error: {e}"); return

        interval_duration_ms = self._interval_to_milliseconds(interval)
        if not interval_duration_ms: logger.error(f"Invalid interval {interval}."); return

        last_saved_open_time_ms = self._get_latest_saved_opentime(symbol, interval)
        next_fetch_start_time_ms = user_start_ts_ms
        if last_saved_open_time_ms is not None:
            calculated_resume_time_ms = last_saved_open_time_ms + interval_duration_ms
            next_fetch_start_time_ms = max(user_start_ts_ms, calculated_resume_time_ms)
            logger.info(f"Resuming for {symbol}/{interval}. Last saved: {last_saved_open_time_ms}, resume calc: {calculated_resume_time_ms}, user start: {user_start_ts_ms}. Effective start: {next_fetch_start_time_ms}")
        else:
            logger.info(f"No prior data for {symbol}/{interval}. Starting from: {user_start_ts_ms}")

        current_loop_start_time_ms = next_fetch_start_time_ms
        total_klines_session = 0
        while current_loop_start_time_ms < target_end_ts_ms:
            fetch_start_dt_str = datetime.fromtimestamp(current_loop_start_time_ms/1000, tz=timezone.utc).strftime(datetime_format)
            logger.info(f"Fetching batch for {symbol}/{interval} from {fetch_start_dt_str} UTC...")
            klines = self.get_klines(symbol, interval, start_time_ms=current_loop_start_time_ms, limit=1000)
            if not klines:
                logger.info(f"No data for {symbol}/{interval} from {fetch_start_dt_str}. End of period or data issue."); break

            klines = [k for k in klines if k['open_time'] < target_end_ts_ms]
            if not klines: logger.info(f"All klines beyond target_end_ts_ms for {symbol}/{interval}."); break

            if self.save_klines_to_csv(klines, symbol, interval):
                total_klines_session += len(klines)
                last_fetched_kline_open_time = klines[-1]['open_time']
                current_loop_start_time_ms = last_fetched_kline_open_time + interval_duration_ms
                if current_loop_start_time_ms >= target_end_ts_ms:
                    logger.info(f"Target end reached for {symbol}/{interval}. Last kline: {last_fetched_kline_open_time}"); break
            else:
                logger.error(f"Save failed for {symbol}/{interval}, batch from {fetch_start_dt_str}. Stopping."); break
            time.sleep(0.5)
        logger.info(f"Fetch complete for {symbol}/{interval}. Fetched in session: {total_klines_session}.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    try:
        fetcher = BinanceDataFetcher()
        logger.info("BinanceDataFetcher instance created.")
        logger.info("Test: fetch_and_save_historical_data for BTCUSDT (1d)...")
        fetcher.fetch_and_save_historical_data(
            symbol="BTCUSDT", interval="1d",
            start_date_str="2023-12-01", end_date_str="2023-12-05"
        )
    except ValueError as e:
        logger.error(f"Init/Config error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in __main__: {e}", exc_info=True)

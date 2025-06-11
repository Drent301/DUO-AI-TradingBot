# core/market_data_fetcher.py
import json
import logging
import os
from typing import Dict, Any, Optional, List
import requests
import csv
import time # Added for sleep
from datetime import datetime, timezone # Added for date parsing and UTC


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.json')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'binance')


class BinanceDataFetcher:
    """
    Fetches historical market data from Binance and can save it to CSV,
    including logic for data resumption.
    """

    def __init__(self):
        self.api_url = self._load_api_url()
        if not self.api_url:
            logger.error("Binance API URL not found in config. Cannot initialize BinanceDataFetcher.")
            raise ValueError("Binance API URL not configured")
        logger.info(f"BinanceDataFetcher initialized with API URL: {self.api_url}")

    def _load_api_url(self) -> Optional[str]:
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config = json.load(f)
            return config.get("data_sources", {}).get("binance", {}).get("api_url")
        except FileNotFoundError:
            logger.error(f"Configuration file not found at {CONFIG_FILE_PATH}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from configuration file: {CONFIG_FILE_PATH}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading API URL: {e}")
        return None

    def _interval_to_milliseconds(self, interval: str) -> Optional[int]:
        """Converts Binance interval string (e.g., '1m', '5m', '1h', '1d') to milliseconds."""
        multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000} # Approx. for M
        try:
            unit = interval[-1]
            value = int(interval[:-1])
            if unit in multipliers:
                return value * multipliers[unit] * 1000
            else:
                logger.error(f"Unsupported interval unit: {unit} in interval {interval}")
                return None
        except Exception as e:
            logger.error(f"Error converting interval '{interval}' to milliseconds: {e}")
            return None

    def _get_latest_saved_opentime(self, symbol: str, interval: str) -> Optional[int]:
        """
        Finds the open_time of the last saved kline for a given symbol and interval.
        """
        directory_path = os.path.join(DATA_DIR, symbol.upper(), interval)
        if not os.path.exists(directory_path):
            logger.info(f"No data directory found for {symbol}/{interval} at {directory_path}. No prior data.")
            return None

        latest_file_start_time = -1
        target_file = None

        try:
            for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):
                    parts = filename.replace(".csv", "").split('_')
                    if len(parts) >= 3: # SYMBOL_INTERVAL_TIMESTAMP
                        try:
                            file_start_time = int(parts[-1])
                            if file_start_time > latest_file_start_time:
                                latest_file_start_time = file_start_time
                                target_file = os.path.join(directory_path, filename)
                        except ValueError:
                            logger.warning(f"Could not parse timestamp from filename: {filename}")
                            continue

            if not target_file:
                logger.info(f"No CSV files found or parsable in {directory_path} for {symbol}/{interval}.")
                return None

            logger.info(f"Identified latest batch file by name: {target_file}")

            # Read this file to find the maximum open_time
            max_open_time_in_file = -1
            with open(target_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row_number, row in enumerate(reader): # Use enumerate for better logging if needed
                    try:
                        open_time = int(row['open_time'])
                        if open_time > max_open_time_in_file:
                            max_open_time_in_file = open_time
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Skipping row in {target_file} due to parsing error: {e} - Row content: {row}")
                        continue

            if max_open_time_in_file != -1:
                logger.info(f"Latest saved open_time found in {target_file} is {max_open_time_in_file}.")
                return max_open_time_in_file
            else:
                logger.info(f"No valid open_time data found in the latest file: {target_file}.")
                return None

        except Exception as e:
            logger.error(f"Error getting latest saved open time for {symbol}/{interval}: {e}")
            return None

    def get_klines(self, symbol: str, interval: str, start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None, limit: int = 500) -> List[Dict[str, Any]]:
        endpoint = f"{self.api_url}/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)
        }
        if start_time_ms:
            params["startTime"] = start_time_ms
        if end_time_ms:
            params["endTime"] = end_time_ms

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
            logger.error(f"HTTP error: {http_err} - {response.status_code} {response.text} for {symbol} with params {params}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error for {symbol}: {req_err}")
        except ValueError as json_err: # Includes json.JSONDecodeError
            logger.error(f"JSON decode error for {symbol}: {json_err}")
        except Exception as e:
            logger.error(f"Unexpected error fetching klines for {symbol}: {e}")
        return []

    def save_klines_to_csv(self, klines: List[Dict[str, Any]], symbol: str, interval: str) -> Optional[str]:
        if not klines:
            logger.warning(f"No k-line data for {symbol}/{interval} to save.")
            return None
        try:
            directory_path = os.path.join(DATA_DIR, symbol.upper(), interval)
            os.makedirs(directory_path, exist_ok=True)
            first_kline_open_time = klines[0]['open_time']
            filename = f"{symbol.upper()}_{interval}_{first_kline_open_time}.csv"
            filepath = os.path.join(directory_path, filename)
            fieldnames = klines[0].keys()
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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

        date_format = "%Y-%m-%d"
        datetime_format = "%Y-%m-%d %H:%M:%S"

        try:
            if ' ' in start_date_str:
                start_dt_obj = datetime.strptime(start_date_str, datetime_format)
            else:
                start_dt_obj = datetime.strptime(start_date_str, date_format)
            user_start_ts_ms = int(start_dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError as e:
            logger.error(f"Invalid start_date_str format: {start_date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Error: {e}")
            return

        if end_date_str:
            try:
                if ' ' in end_date_str:
                    end_dt_obj = datetime.strptime(end_date_str, datetime_format)
                else:
                    end_dt_obj = datetime.strptime(end_date_str, date_format)
                target_end_ts_ms = int(end_dt_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError as e:
                logger.error(f"Invalid end_date_str format: {end_date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS. Error: {e}")
                return
        else:
            target_end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        interval_duration_ms = self._interval_to_milliseconds(interval)
        if not interval_duration_ms:
            logger.error(f"Invalid interval {interval}, cannot proceed.")
            return

        last_saved_open_time_ms = self._get_latest_saved_opentime(symbol, interval)

        next_fetch_start_time_ms = user_start_ts_ms
        if last_saved_open_time_ms is not None:
            calculated_resume_time_ms = last_saved_open_time_ms + interval_duration_ms
            next_fetch_start_time_ms = max(user_start_ts_ms, calculated_resume_time_ms)
            logger.info(f"Resuming download for {symbol}/{interval}. Last saved open: {last_saved_open_time_ms}, calculated resume: {calculated_resume_time_ms}, user start: {user_start_ts_ms}. Effective start: {next_fetch_start_time_ms}")
        else:
            logger.info(f"No prior data found for {symbol}/{interval}. Starting fetch from user_start_ts_ms: {user_start_ts_ms}")

        current_loop_start_time_ms = next_fetch_start_time_ms
        total_klines_fetched_session = 0

        while current_loop_start_time_ms < target_end_ts_ms:
            fetch_start_dt = datetime.fromtimestamp(current_loop_start_time_ms/1000, tz=timezone.utc).strftime(datetime_format)
            logger.info(f"Fetching batch for {symbol}/{interval} starting from {fetch_start_dt} (UTC)...")

            klines = self.get_klines(symbol, interval, start_time_ms=current_loop_start_time_ms, limit=1000)

            if not klines:
                logger.info(f"No more data returned for {symbol}/{interval} from start time {fetch_start_dt}. Assuming end of available data for this period.")
                break

            # Filter klines that are beyond the target_end_ts_ms (if any)
            klines = [k for k in klines if k['open_time'] < target_end_ts_ms]
            if not klines:
                logger.info(f"All fetched klines were beyond target_end_ts_ms. Stopping for {symbol}/{interval}.")
                break

            saved_path = self.save_klines_to_csv(klines, symbol, interval)
            if saved_path:
                total_klines_fetched_session += len(klines)
                last_fetched_kline_open_time = klines[-1]['open_time']
                current_loop_start_time_ms = last_fetched_kline_open_time + interval_duration_ms

                if current_loop_start_time_ms >= target_end_ts_ms:
                    logger.info(f"Target end date reached for {symbol}/{interval}. Last kline open_time: {last_fetched_kline_open_time}")
                    break
            else:
                logger.error(f"Failed to save klines for {symbol}/{interval}, batch starting {fetch_start_dt}. Stopping fetch for this symbol/interval to avoid data loss.")
                break

            logger.info(f"Politely sleeping for 0.5 seconds before next API call...")
            time.sleep(0.5)

        logger.info(f"Historical data fetch process completed for {symbol}/{interval}. Total klines fetched in this session: {total_klines_fetched_session}.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    try:
        fetcher = BinanceDataFetcher()
        logger.info("BinanceDataFetcher instance created successfully.")

        # Test fetch_and_save_historical_data
        # Ensure DATA_DIR and its subdirectories are writable by the script
        # Example: Fetch 2 days of 1d data for BTCUSDT
        # To test resumption:
        # 1. Run once: fetcher.fetch_and_save_historical_data(symbol="BTCUSDT", interval="1d", start_date_str="2023-12-01", end_date_str="2023-12-03")
        # 2. Check CSV created.
        # 3. Run again, extending the end date: fetcher.fetch_and_save_historical_data(symbol="BTCUSDT", interval="1d", start_date_str="2023-12-01", end_date_str="2023-12-05")
        #    It should resume from where it left off (e.g., start fetching from 2023-12-03 + 1 day).

        logger.info("Starting test of fetch_and_save_historical_data for BTCUSDT...")
        fetcher.fetch_and_save_historical_data(
            symbol="BTCUSDT",
            interval="1d",
            start_date_str="2023-12-01",
            end_date_str="2023-12-05"
        )

        # Example for a shorter interval, e.g., 1h for a few hours
        # logger.info("Starting test of fetch_and_save_historical_data for ETHUSDT (1h)...")
        # fetcher.fetch_and_save_historical_data(
        #     symbol="ETHUSDT",
        #     interval="1h",
        #     start_date_str="2023-12-01 00:00:00", # More specific start time
        #     end_date_str="2023-12-01 05:00:00"   # Fetch 5 hours of data
        # )

    except ValueError as e:
        logger.error(f"Failed during BinanceDataFetcher operations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during __main__ test: {e}", exc_info=True)

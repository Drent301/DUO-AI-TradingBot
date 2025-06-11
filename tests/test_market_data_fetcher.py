import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
import time # For checking time.sleep calls if necessary
from datetime import datetime, timezone # For timestamp conversions

# Adjust path to import BinanceDataFetcher
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.market_data_fetcher import BinanceDataFetcher, DATA_DIR # Assuming DATA_DIR is exposed or accessible for testing paths

# Helper to create dummy config content
def get_dummy_config(include_binance_url=True):
    config = {
        "data_sources": {
            "binance": {}
        }
    }
    if include_binance_url:
        config["data_sources"]["binance"]["api_url"] = "https://api.testbinance.com"
    return json.dumps(config)

class TestBinanceDataFetcher(unittest.TestCase):

    def setUp(self):
        # Ensure DATA_DIR for tests is unique if needed, or clean up
        self.test_data_dir = os.path.join(DATA_DIR, "test_symbol", "test_interval")
        # os.makedirs(self.test_data_dir, exist_ok=True) # Not always needed if mocking os.makedirs

        # Mock config file path used by BinanceDataFetcher's _load_api_url
        self.mock_config_path = 'core.market_data_fetcher.CONFIG_FILE_PATH'


    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_initialization_success(self, mock_json_load, mock_file_open):
        mock_json_load.return_value = json.loads(get_dummy_config(include_binance_url=True))
        # Patch CONFIG_FILE_PATH directly within the scope of the test
        with patch(self.mock_config_path, "dummy/path/config.json"):
            fetcher = BinanceDataFetcher()
        self.assertEqual(fetcher.api_url, "https://api.testbinance.com")
        # The actual call to open uses the value of CONFIG_FILE_PATH at the time of _load_api_url call
        mock_file_open.assert_called_once_with("dummy/path/config.json", 'r')


    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_initialization_no_url(self, mock_json_load, mock_file_open):
        mock_json_load.return_value = json.loads(get_dummy_config(include_binance_url=False))
        with patch(self.mock_config_path, "dummy/path/config.json"):
            with self.assertRaises(ValueError) as context:
                BinanceDataFetcher()
        self.assertIn("Binance API URL not configured", str(context.exception))

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_initialization_config_not_found(self, mock_file_open):
        with patch(self.mock_config_path, "dummy/path/config.json"):
            with self.assertRaises(ValueError) as context:
                BinanceDataFetcher()
        self.assertIn("Binance API URL not configured", str(context.exception))

    def test_interval_to_milliseconds(self):
        with patch(self.mock_config_path, "dummy/path/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())):
            fetcher = BinanceDataFetcher()
        self.assertEqual(fetcher._interval_to_milliseconds('1m'), 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('5m'), 5 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1h'), 60 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1d'), 24 * 60 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('3d'), 3 * 24 * 60 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1w'), 7 * 24 * 60 * 60 * 1000)
        # As per implementation using 30 days for 1M
        self.assertEqual(fetcher._interval_to_milliseconds('1M'), 30 * 24 * 60 * 60 * 1000)
        self.assertIsNone(fetcher._interval_to_milliseconds('1x'))
        self.assertIsNone(fetcher._interval_to_milliseconds('m1'))


    @patch('requests.get')
    def test_get_klines_success(self, mock_requests_get):
        with patch(self.mock_config_path, "dummy/path/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())):
            fetcher = BinanceDataFetcher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1609459200000, "100.0", "110.0", "90.0", "105.0", "1000.0", 1609459259999, "105000.0", 100, "500.0", "52500.0", "0"],
            [1609459260000, "105.0", "115.0", "95.0", "110.0", "1200.0", 1609459319999, "126000.0", 120, "600.0", "63000.0", "0"]
        ]
        mock_requests_get.return_value = mock_response

        klines = fetcher.get_klines("BTCUSDT", "1m", limit=2)
        self.assertEqual(len(klines), 2)
        self.assertEqual(klines[0]['open'], 100.0)
        self.assertEqual(klines[1]['volume'], 1200.0)
        mock_requests_get.assert_called_once()
        args, kwargs = mock_requests_get.call_args
        self.assertTrue(args[0].startswith("https://api.testbinance.com"))
        self.assertIn("/klines", args[0])
        self.assertEqual(kwargs['params']['symbol'], "BTCUSDT")
        self.assertEqual(kwargs['params']['limit'], 2)

    @patch('requests.get')
    def test_get_klines_http_error(self, mock_requests_get):
        with patch(self.mock_config_path, "dummy/path/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())):
            fetcher = BinanceDataFetcher()

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test HTTP Error", response=mock_response)
        mock_requests_get.return_value = mock_response

        klines = fetcher.get_klines("BTCUSDT", "1m")
        self.assertEqual(len(klines), 0)

    @patch('os.makedirs')
    def test_save_klines_to_csv_success(self, mock_os_makedirs):
        # Mock open for __init__ and for the CSV writing part separately
        with patch(self.mock_config_path, "dummy/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())) as mock_init_open:
            fetcher = BinanceDataFetcher()

        klines_data = [
            {"open_time": 1609459200000, "open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1000.0, "close_time": 1609459259999, "quote_asset_volume": 105000.0, "number_of_trades": 100, "taker_buy_base_asset_volume": 500.0, "taker_buy_quote_asset_volume": 52500.0}
        ]
        symbol = "TESTBTC"
        interval = "1d"

        m_open_csv = mock_open()
        with patch('builtins.open', m_open_csv): # This mock will be used by save_klines_to_csv
            file_path = fetcher.save_klines_to_csv(klines_data, symbol, interval)

        expected_dir = os.path.join(DATA_DIR, symbol.upper(), interval)
        mock_os_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

        expected_filename = f"{symbol.upper()}_{interval}_{klines_data[0]['open_time']}.csv"
        self.assertIsNotNone(file_path)
        self.assertIn(expected_filename, file_path)

        m_open_csv.assert_called_once_with(os.path.join(expected_dir, expected_filename), 'w', newline='', encoding='utf-8')
        handle = m_open_csv()
        # Check header write based on keys of first kline
        self.assertEqual(handle.write.call_args_list[0], call(','.join(klines_data[0].keys()) + '\r\n'))


    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_latest_saved_opentime(self, mock_os_listdir, mock_os_path_exists):
        with patch(self.mock_config_path, "dummy/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())) as mock_init_open:
            fetcher = BinanceDataFetcher()

        symbol = "TESTLATEST"
        interval = "1h"
        dir_path = os.path.join(DATA_DIR, symbol.upper(), interval)

        mock_os_path_exists.return_value = True
        mock_os_listdir.return_value = [f"{symbol.upper()}_{interval}_1600000000000.csv", f"{symbol.upper()}_{interval}_1700000000000.csv"]

        csv_content_latest = "open_time,open,close\n1700000000000,10,11\n1700003600000,11,12"
        csv_content_older = "open_time,open,close\n1600000000000,1,2"

        # Mock open to return different content based on filename
        def mock_open_side_effect(path, mode, newline='', encoding=''):
            if str(path).endswith("1700000000000.csv"):
                return mock_open(read_data=csv_content_latest)()
            elif str(path).endswith("1600000000000.csv"):
                return mock_open(read_data=csv_content_older)()
            return mock_open(read_data="")() # Default for any other call

        with patch('builtins.open', side_effect=mock_open_side_effect):
            latest_time = fetcher._get_latest_saved_opentime(symbol, interval)

        self.assertEqual(latest_time, 1700003600000)
        mock_os_listdir.assert_called_once_with(dir_path)


    @patch('core.market_data_fetcher.BinanceDataFetcher._get_latest_saved_opentime')
    @patch('core.market_data_fetcher.BinanceDataFetcher.get_klines')
    @patch('core.market_data_fetcher.BinanceDataFetcher.save_klines_to_csv')
    @patch('time.sleep', return_value=None)
    def test_fetch_and_save_historical_data_no_existing_data(self, mock_time_sleep, mock_save_csv, mock_get_klines, mock_get_latest_time):
        with patch(self.mock_config_path, "dummy/config.json"), \
             patch('builtins.open', mock_open(read_data=get_dummy_config())) as mock_init_open:
            fetcher = BinanceDataFetcher()

        mock_get_latest_time.return_value = None

        start_ts = int(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        interval_ms = 3600000 # 1h

        klines_batch1 = [{'open_time': start_ts + i*interval_ms, 'close': i+1} for i in range(3)]
        klines_batch2 = [{'open_time': start_ts + (i+3)*interval_ms, 'close': i+10} for i in range(2)]

        mock_get_klines.side_effect = [klines_batch1, klines_batch2, []]
        mock_save_csv.return_value = "dummy_path.csv"

        fetcher.fetch_and_save_historical_data("MYSYMBOL", "1h", "2023-01-01 00:00:00", "2023-01-01 05:00:00")

        self.assertEqual(mock_get_klines.call_count, 3)
        self.assertEqual(mock_save_csv.call_count, 2)

        self.assertEqual(mock_get_klines.call_args_list[0][1]['start_time_ms'], start_ts)

        expected_second_start_ts_ms = klines_batch1[-1]['open_time'] + interval_ms
        self.assertEqual(mock_get_klines.call_args_list[1][1]['start_time_ms'], expected_second_start_ts_ms)

        self.assertEqual(mock_time_sleep.call_count, 2)


if __name__ == '__main__':
    # Create 'tests' directory if it doesn't exist
    if not os.path.exists('tests'):
        os.makedirs('tests', exist_ok=True)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

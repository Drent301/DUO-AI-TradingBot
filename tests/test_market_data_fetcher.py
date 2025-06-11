import unittest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
import time
from datetime import datetime, timezone
import tempfile # Added
import shutil # Added (though TemporaryDirectory cleanup is usually enough)

# Adjust path to import BinanceDataFetcher
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.market_data_fetcher import BinanceDataFetcher
# Removed DATA_DIR import as tests will use temporary directories managed by fetcher.data_dir

class TestBinanceDataFetcher(unittest.TestCase):

    def setUp(self):
        self.test_temp_dir = tempfile.TemporaryDirectory()
        self.mock_config_path_str = 'core.market_data_fetcher.CONFIG_FILE_PATH'

        # Base dummy config content as a dictionary
        self.dummy_config_dict = {
            "data_sources": {
                "binance": {
                    "api_url": "https://api.testbinance.com"
                    # "data_directory" will be added per test if needed
                }
            }
        }

    def tearDown(self):
        self.test_temp_dir.cleanup()

    def _get_fetcher_with_mocked_config(self, config_dict_to_use):
        """Helper to initialize fetcher with a specific mocked config."""
        with patch(self.mock_config_path_str, "dummy/config.json"), \
             patch('builtins.open', mock_open(read_data=json.dumps(config_dict_to_use))) as mock_file, \
             patch('json.load', return_value=config_dict_to_use) as mock_json:
            return BinanceDataFetcher()

    def test_initialization_success_with_data_dir_in_config(self):
        config_data = self.dummy_config_dict.copy()
        custom_data_path = os.path.join(self.test_temp_dir.name, "custom_data")
        config_data["data_sources"]["binance"]["data_directory"] = custom_data_path

        fetcher = self._get_fetcher_with_mocked_config(config_data)

        self.assertEqual(fetcher.api_url, "https://api.testbinance.com")
        self.assertEqual(fetcher.data_dir, os.path.abspath(custom_data_path))

    def test_initialization_default_data_dir(self):
        # Test default data_dir when not specified in config
        # Mock CONFIG_FILE_PATH to control the base for default path calculation
        mock_project_root = os.path.abspath(os.path.join(self.test_temp_dir.name, "mock_project_root"))
        mock_config_file = os.path.join(mock_project_root, "config", "config.json")
        expected_default_data_dir = os.path.join(mock_project_root, "data", "binance")

        with patch(self.mock_config_path_str, mock_config_file):
            # Ensure _load_config returns a config without data_directory
            config_without_data_dir = self.dummy_config_dict.copy()
            if "data_directory" in config_without_data_dir["data_sources"]["binance"]:
                del config_without_data_dir["data_sources"]["binance"]["data_directory"]

            fetcher = self._get_fetcher_with_mocked_config(config_without_data_dir)
            self.assertEqual(fetcher.data_dir, expected_default_data_dir)


    def test_initialization_no_url(self):
        config_no_url = {"data_sources": {"binance": {}}} # Missing api_url
        with self.assertRaises(ValueError) as context:
            self._get_fetcher_with_mocked_config(config_no_url)
        self.assertIn("Binance API URL not configured", str(context.exception))

    def test_initialization_config_not_found(self):
        # Patch open to raise FileNotFoundError for the _load_config call
        with patch(self.mock_config_path_str, "dummy/nonexistent_config.json"), \
             patch('builtins.open', side_effect=FileNotFoundError):
            with self.assertRaises(ValueError) as context:
                BinanceDataFetcher() # Directly call to test __init__ path for _load_config returning None
        self.assertIn("Failed to load configuration", str(context.exception))


    def test_interval_to_milliseconds(self):
        fetcher = self._get_fetcher_with_mocked_config(self.dummy_config_dict)
        self.assertEqual(fetcher._interval_to_milliseconds('1m'), 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1h'), 60 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1d'), 24 * 60 * 60 * 1000)
        self.assertEqual(fetcher._interval_to_milliseconds('1M'), 30 * 24 * 60 * 60 * 1000) # Approx based on impl.
        self.assertIsNone(fetcher._interval_to_milliseconds('1x'))

    @patch('requests.get')
    def test_get_klines_success(self, mock_requests_get):
        fetcher = self._get_fetcher_with_mocked_config(self.dummy_config_dict)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1609459200000, "100.0", "110.0", "90.0", "105.0", "1000.0", 1609459259999, "105000.0", 100, "500.0", "52500.0", "0"]
        ]
        mock_requests_get.return_value = mock_response
        klines = fetcher.get_klines("BTCUSDT", "1m", limit=1)
        self.assertEqual(len(klines), 1)
        self.assertEqual(klines[0]['open'], 100.0)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open) # This will mock open for the CSV writing
    def test_save_klines_to_csv_success(self, mock_csv_write_open, mock_os_makedirs):
        config_data = self.dummy_config_dict.copy()
        test_specific_data_path = os.path.join(self.test_temp_dir.name, "csv_data")
        config_data["data_sources"]["binance"]["data_directory"] = test_specific_data_path

        fetcher = self._get_fetcher_with_mocked_config(config_data)

        # Ensure klines_data matches KLINE_CSV_FIELDNAMES
        klines_data = [{key: (idx if "time" in key or "trade" in key else float(idx * 10) + 0.5)
                        for idx, key in enumerate(BinanceDataFetcher.KLINE_CSV_FIELDNAMES)}]

        symbol = "TESTSAVE"
        interval = "1m"

        file_path = fetcher.save_klines_to_csv(klines_data, symbol, interval)

        expected_dir = os.path.join(test_specific_data_path, symbol.upper(), interval)
        mock_os_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

        expected_filename = f"{symbol.upper()}_{interval}_{klines_data[0]['open_time']}.csv"
        self.assertEqual(file_path, os.path.join(expected_dir, expected_filename))

        mock_csv_write_open.assert_called_once_with(file_path, 'w', newline='', encoding='utf-8')

        file_handle_mock = mock_csv_write_open()
        self.assertTrue(file_handle_mock.write.called)
        actual_header_written = file_handle_mock.write.call_args_list[0].args[0]
        expected_header_string = ','.join(BinanceDataFetcher.KLINE_CSV_FIELDNAMES) + os.linesep
        self.assertEqual(actual_header_written, expected_header_string)

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open) # Mocks open for reading CSV in _get_latest_saved_opentime
    def test_get_latest_saved_opentime(self, mock_csv_read_open, mock_os_listdir, mock_os_path_exists):
        config_data = self.dummy_config_dict.copy()
        test_specific_data_path = os.path.join(self.test_temp_dir.name, "latest_time_data")
        config_data["data_sources"]["binance"]["data_directory"] = test_specific_data_path
        fetcher = self._get_fetcher_with_mocked_config(config_data)

        symbol = "TESTLATEST"
        interval = "1h"
        # This is now fetcher.data_dir due to config mocking
        dir_path = os.path.join(test_specific_data_path, symbol.upper(), interval)

        mock_os_path_exists.return_value = True
        mock_os_listdir.return_value = [f"{symbol.upper()}_{interval}_1700000000000.csv"]

        csv_content = "open_time,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume\n" \
                      "1700000000000,10,11,9,10.5,100,1700003599999,1050,10,50,525\n" \
                      "1700003600000,10.5,12,10,11.5,120,1700007199999,1250,12,60,625"
        # mock_csv_read_open needs to be the one used by _get_latest_saved_opentime
        mock_csv_read_open.return_value = mock_open(read_data=csv_content).return_value

        latest_time = fetcher._get_latest_saved_opentime(symbol, interval)

        self.assertEqual(latest_time, 1700003600000)
        mock_os_listdir.assert_called_once_with(dir_path)
        mock_csv_read_open.assert_called_once_with(os.path.join(dir_path, f"{symbol.upper()}_{interval}_1700000000000.csv"), 'r', newline='', encoding='utf-8')


    @patch('core.market_data_fetcher.BinanceDataFetcher._get_latest_saved_opentime')
    @patch('core.market_data_fetcher.BinanceDataFetcher.get_klines')
    @patch('core.market_data_fetcher.BinanceDataFetcher.save_klines_to_csv')
    @patch('time.sleep', return_value=None)
    def test_fetch_and_save_historical_data_no_existing_data(self, mock_time_sleep, mock_save_csv, mock_get_klines, mock_get_latest_time):
        config_data = self.dummy_config_dict.copy()
        test_specific_data_path = os.path.join(self.test_temp_dir.name, "fetch_hist_data")
        config_data["data_sources"]["binance"]["data_directory"] = test_specific_data_path
        fetcher = self._get_fetcher_with_mocked_config(config_data)

        mock_get_latest_time.return_value = None
        start_ts = int(datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        interval_ms = 3600000 # 1h

        # Ensure klines_data matches KLINE_CSV_FIELDNAMES structure
        def create_mock_kline(ts_offset, base_val):
            kline = {key: 0 for key in BinanceDataFetcher.KLINE_CSV_FIELDNAMES}
            kline.update({
                'open_time': start_ts + ts_offset * interval_ms,
                'close': float(base_val + ts_offset),
                'open': float(base_val + ts_offset -1),
                'high': float(base_val + ts_offset +1),
                'low': float(base_val + ts_offset -2),
                'volume': float(1000 + ts_offset*10),
                'close_time': start_ts + ts_offset * interval_ms + interval_ms -1,
                'number_of_trades': 100 + ts_offset,
                'quote_asset_volume': float( (base_val+ts_offset) * (1000+ts_offset*10)),
                'taker_buy_base_asset_volume': float(500+ts_offset*5),
                'taker_buy_quote_asset_volume': float((base_val+ts_offset) * (500+ts_offset*5))
            })
            return kline

        klines_batch1 = [create_mock_kline(i, 1) for i in range(3)]
        klines_batch2 = [create_mock_kline(i + 3, 10) for i in range(2)]

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

# Data Management

This document describes the scripts and procedures for downloading and validating historical market data for the trading bot.

## Data Downloading

The `utils/data_downloader.py` script is used to download historical market data using the Freqtrade command-line interface.

**Functionality:**
- Downloads OHLCV (Open, High, Low, Close, Volume) data for specified currency pairs and timeframes.
- Connects to a specified exchange (e.g., Binance).
- Can download data for a defined period (e.g., the last 5 years).
- Stores data in the `user_data/data/<exchange_name>/` directory in JSON format, as expected by Freqtrade.

**Usage:**
The script can be run directly or, more conveniently, via `run_pipeline.py`:

```bash
python run_pipeline.py --download-data
```
This command will download data for the predefined pairs and timeframes:
- Pairs: ZEN/BTC, LSK/BTC, ETH/BTC, ETH/EUR
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Exchange: Binance
- Period: 5 years

The script assumes `freqtrade` is installed and configured in the system's PATH.

## Data Validation

The `utils/data_validator.py` script is used to validate the downloaded historical data.

**Functionality:**
- Loads the downloaded JSON data files for specified pairs and timeframes.
- Verifies the presence and integrity of OHLCV data.
- Attempts to calculate common technical indicators (RSI, MACD) using `ta-lib` to ensure data compatibility.
- Provides a basic check and descriptive information regarding data suitability for CNN model input, referencing `core/cnn_patterns.py`.

**Usage:**
The script can be run directly or, more conveniently, via `run_pipeline.py`:

```bash
python run_pipeline.py --validate-data
```
This command will validate the data for the same predefined pairs and timeframes downloaded by the `--download-data` command, assuming the data exists in `user_data/data/binance/`.

The script will print output detailing the validation checks for each file, including:
- OHLCV data presence and basic integrity.
- Success or failure of indicator calculations.
- Information regarding CNN data suitability.

---

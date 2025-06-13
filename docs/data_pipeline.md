# Data Pipeline for CNN Model Pre-training

## Overview

The pre-training pipeline is designed to automate the collection of historical market data, process this data to generate relevant features and pattern-based labels, and then train Convolutional Neural Network (CNN) models for various identified patterns. This process aims to create a foundational set of models that can later be used in a live trading environment to detect similar patterns.

## Configuration

The pipeline's behavior is primarily configured through parameters managed by `core/params_manager.py` (persisted in `memory/learned_params.json`), a market regime definition file, and environment variables.

### 1. `core/params_manager.py` (`memory/learned_params.json`)

This module manages global and strategy-specific parameters. For the data pipeline, key **global parameters** include:

*   **Data Fetching:**
    *   `data_fetch_start_date_str`: (String "YYYY-MM-DD") The default start date for fetching historical data if no specific market regimes are used or if fetching a global range.
    *   `data_fetch_end_date_str`: (String "YYYY-MM-DD") The default end date. If empty, data is fetched up to the latest available.
    *   `data_fetch_pairs`: (List of strings) Trading pairs to fetch data for (e.g., `["ETH/EUR", "BTC/USDT"]`).
    *   `data_fetch_timeframes`: (List of strings) Timeframes to fetch for each pair (e.g., `["1h", "4h"]`).
    *   `patterns_to_train`: (List of strings) Specifies which pattern types the pipeline should prepare data for and train models on (e.g., `["bullFlag", "bearishEngulfing"]`).

*   **Labeling Parameters (`pattern_labeling_configs`):**
    *   This is a dictionary where each key is a `pattern_type` string (e.g., "bullFlag", "bearishEngulfing").
    *   The value for each pattern type is another dictionary containing:
        *   `future_N_candles`: (Integer) How many candles into the future to look for the pattern's outcome.
        *   `profit_threshold_pct`: (Float) The percentage change defining a "profitable" outcome. For bullish patterns, this is a price increase. For bearish patterns, this is a price decrease.
        *   `loss_threshold_pct`: (Float) The percentage change defining a "loss" or failed outcome (i.e., price moving significantly against the pattern's expected direction). For bullish patterns, this is typically a negative value (price decrease). For bearish patterns, this is also typically a negative value (representing an adverse price increase).
    *   Example:
      ```json
      "pattern_labeling_configs": {
          "bullFlag": {
              "future_N_candles": 20,
              "profit_threshold_pct": 0.02,
              "loss_threshold_pct": -0.01
          },
          "bearishEngulfing": {
              "future_N_candles": 20,
              "profit_threshold_pct": 0.02, // e.g., a 2% drop is profit
              "loss_threshold_pct": -0.01  // e.g., a 1% rise is a loss for the pattern
          }
      }
      ```

### 2. `data/market_regimes.json`

This JSON file allows for defining specific historical periods for data collection, tailored to different market conditions (regimes) for each trading pair. This enables the creation of more diverse datasets.

*   **Structure:** A dictionary where keys are trading pairs (e.g., "ETH/EUR").
    *   Each pair contains categories like `high_volatility`, `low_volatility`, `bull_trend`, `bear_trend`.
    *   Each category is a list of `{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}` objects.
*   **Purpose:** If regimes are defined for a pair, the pipeline will fetch data for these specific periods. If no regimes are defined for a pair, or if the regime list is empty, it falls back to using the global `data_fetch_start_date_str` and `data_fetch_end_date_str` from `params_manager`.

### 3. `.env` File

This file stores sensitive credentials and environment-specific configurations. For the data pipeline:
*   `BITVAVO_API_KEY`: Your API key for the Bitvavo exchange.
*   `BITVAVO_SECRET_KEY`: Your API secret for the Bitvavo exchange.
    *These are essential for fetching live data from Bitvavo.*

## Data Flow

1.  **Fetching:**
    *   The `PreTrainer` module in `core/pre_trainer.py` orchestrates the process.
    *   It iterates through specified pairs, timeframes, and patterns.
    *   For each combination, `fetch_historical_data` is called.
    *   This method first checks `data/market_regimes.json`. If defined periods exist for the symbol, data for these specific periods is fetched.
    *   If no regimes are specified for the symbol or if they yield no data, it falls back to fetching data based on the global `data_fetch_start_date_str` and `data_fetch_end_date_str` from `params_manager`.
    *   Data is fetched from Bitvavo via `BitvavoExecutor`, which uses the `ccxt` library.

2.  **Caching:**
    *   Fetched raw OHLCV data (for specific periods, either from regimes or global range) is cached to avoid redundant API calls.
    *   Cache files are stored as CSVs in: `data/raw_historical_data/{symbol_sanitized}/{timeframe}/{symbol_sanitized}_{timeframe}_{start_timestamp}_{end_timestamp}.csv`.
    *   On subsequent requests for the exact same period, data is read from the cache.

3.  **Processing (`prepare_training_data`):**
    *   For each fetched dataset (per pair/timeframe combination), and for each `pattern_type` to be trained:
        *   **Feature Engineering:** Common technical indicators (RSI, MACD, Bollinger Bands) are calculated.
        *   **Pattern-Specific Labeling (Two-Stage Process):**
            1.  **Form-Based Detection:** The system first attempts to identify potential pattern candidates based on their candlestick shapes and/or immediate surrounding indicator values using rules defined in `CNNPatterns` (e.g., `detect_bull_flag`, `_detect_engulfing`). This step marks candles where the basic form of the pattern is present.
            2.  **Outcome-Based Labeling:** For candidates that passed the form-based detection, a future-looking analysis is performed using parameters from `pattern_labeling_configs` (specific to the `pattern_type`). This checks if the price movement over the configured `future_N_candles` met the `profit_threshold_pct` without first hitting the `loss_threshold_pct`.
            *   If both form detection and outcome criteria are met, the pattern is labeled as positive (1) in a column like `bullFlag_label`. Otherwise, it's labeled (0).
            *   Samples not matching the initial form criteria are directly labeled as (0) for that pattern type.

4.  **Training (`train_ai_models`):**
    *   A separate CNN model is trained for each combination of symbol, timeframe, and `pattern_type`.
    *   The `training_data` (output from `prepare_training_data`) and the specific `target_label_column` are used.
    *   The trained model and its associated scaler parameters are saved to disk.

## Output Artifacts

The pipeline produces the following:

*   **Trained Models:**
    *   Path: `data/models/{symbol_sanitized}/{timeframe}/cnn_model_{pattern_type}.pth`
    *   Example: `data/models/ETH_EUR/1h/cnn_model_bullFlag.pth`
*   **Scaler Parameters:**
    *   Path: `data/models/{symbol_sanitized}/{timeframe}/scaler_params_{pattern_type}.json`
    *   Example: `data/models/ETH_EUR/1h/scaler_params_bullFlag.json`
    *   Contains `feature_names_in_`, `min_`, `scale_`, and `sequence_length` for data normalization.
*   **Pre-training Log:**
    *   Path: `memory/pre_train_log.json`
    *   A JSON file logging details of each training run, including model type, data size, performance metrics, and paths to saved artifacts.
*   **Cached Raw Data:**
    *   Path: `data/raw_historical_data/{symbol_sanitized}/{timeframe}/... .csv`
*   **Time Effectiveness Analysis (Optional):**
    *   Path: `memory/time_effectiveness.json` (may be overwritten per pattern type in current setup)

## Running the Pipeline

The main entry point for running the pre-training pipeline is the `if __name__ == "__main__":` block within `core/pre_trainer.py`.
It can typically be executed as a Python module:

```bash
python -m core.pre_trainer
```

This will:
1.  Initialize `ParamsManager` (loading parameters from `memory/learned_params.json` or defaults).
2.  Initialize `PreTrainer`.
3.  For the test configuration within the `__main__` block:
    *   Clear relevant cache and model output directories.
    *   Set specific test parameters (pairs, timeframes, patterns, date ranges) in `ParamsManager`.
    *   Call `pre_trainer.run_pretraining_pipeline()`.
4.  The pipeline then iterates through the configured pairs, timeframes, and patterns, performing fetching, processing, and training.
5.  Test output will indicate created files and provide basic success/failure messages.

Ensure your `.env` file is correctly set up with API keys if you intend to fetch new live data.

# DUO-AI Freqtrade Integration

## Overview

DUO-AI Freqtrade Integration is an advanced system designed to augment Freqtrade with AI-driven decision-making capabilities. It leverages reflective AI, machine learning pattern recognition, and dynamic strategy adjustments to optimize cryptocurrency trading performance. The system is built to learn from its operations, adapt to market changes, and continuously refine its trading strategies.

## Key Features & Components

*   **AI-Powered Strategy Optimization**: Utilizes AI analysis (potentially from models like GPT/Grok, though specific model integration details are part of ongoing development) to reflect on trade performance and propose mutations to Freqtrade strategies.
*   **Dynamic Configuration**: While not hot-swapping Freqtrade's core configuration during live runs, the system can dynamically adjust strategy parameters and potentially advise on filter adjustments based on AI analysis.
*   **Pattern Recognition (`cnn_patterns.py`)**: Identifies potential market patterns. Currently, `cnn_patterns.py` operates on a rule-based system for pattern detection. Future development aims to integrate trained Machine Learning (ML) models for more sophisticated pattern analysis.
*   **Trade Data Analysis**: Primary trade logging and analysis for AI learning now rely on Freqtrade's internal SQLite database (`freqtrade.sqlite`). This database is the source of truth for performance metrics used by the AI components.
*   **Trade Export (`trade_logger.py`)**: The `trade_logger.py` component has been repurposed. Instead of being the primary logger, it will serve as a utility for exporting trade data in various formats for external analysis or backup, complementing the main Freqtrade database.
*   **AI-Specific Cooldowns (`cooldown_tracker.py`)**: Implements a cooldown mechanism for AI-driven actions (like strategy mutations or new advice generation) to prevent rapid, potentially unstable changes and allow strategies to operate for a period before re-evaluation.
*   **Strategy Management (`strategy_manager.py`)**: Manages the parameters and performance tracking of Freqtrade strategies, interfacing with the Freqtrade database and parameter files.
*   **AI Optimization Loop (`ai_optimizer.py`)**: Orchestrates the periodic optimization process, including performance analysis, reflection, and strategy mutation.

## Configuration

### Pair Whitelist

The `pair_whitelist` in your Freqtrade configuration should be carefully selected. The AI system will operate on these pairs. Example:

```json
"exchange": {
    "pair_whitelist": [
        "ETH/USDT",
        "BTC/USDT",
        "ADA/USDT",
        "SOL/USDT",
        "XRP/USDT",
        // Add more pairs as needed, ensure they are valid on your exchange
    ]
    // ... other exchange settings
},
```

### Dynamic Adjustments

The system's approach to dynamic Freqtrade configuration adjustments involves the AI providing advice or suggesting modifications to strategy parameters or filter lists. These are not "hot-swaps" of the main Freqtrade `config.json` during live operations but rather updates to strategy-specific parameters or through controlled mechanisms that Freqtrade supports for dynamic loading (e.g., strategy parameters).

## Critical Testing Advisory

**IMPORTANT**: Before considering any live deployment, this system requires **extensive backtesting and prolonged dry-run testing** on a compatible Freqtrade setup. AI-driven trading strategies can be complex and may introduce unforeseen risks.

*   **Backtesting**: Use Freqtrade's backtesting capabilities to evaluate strategy performance over historical data.
*   **Dry-Run**: Run the bot in a simulated environment (dry-run mode) on a live market to observe its behavior without risking real capital.
*   **Gradual Exposure**: If moving to live trading, start with minimal capital and closely monitor performance and behavior.

## Setup & Installation (Conceptual)

1.  **Freqtrade**: Ensure you have a working Freqtrade installation.
2.  **Python Environment**: Set up a Python environment (e.g., venv) with dependencies from `requirements.txt`.
    ```bash
    python -m venv .venv
    source .venv/bin/activate # or .venv\Scripts\activate for Windows
    pip install -r requirements.txt
    ```
3.  **Configuration**:
    *   Configure Freqtrade (`config.json`) with your exchange details, strategy (e.g., `DUOAI_Strategy`), and whitelisted pairs.
    *   Set up any necessary API keys or environment variables (e.g., in a `.env` file).
4.  **Database**: Ensure Freqtrade is configured to use an SQLite database, as this is used by the AI components for performance analysis.

## Usage (Conceptual)

The primary entry point for the AI optimization logic is `main.py`.

```bash
python main.py
```

This would typically initialize the `AIOptimizer` and start its periodic optimization cycles, interacting with a running Freqtrade instance or its data.

## Fetching Historical Market Data

The project includes a script to download historical k-line (candlestick) data from Binance. This data is stored locally and can be used for backtesting, analysis, or training machine learning models.

The script `scripts/fetch_market_data.py` handles fetching and saving the data, including logic to resume downloads if interrupted.

### How to Run

You can run the script from the project's root directory:

```bash
python scripts/fetch_market_data.py --symbols BTCUSDT,ETHUSDT --intervals 1d,4h --start_date "2020-01-01" --end_date "2023-12-31"
```

### Command-Line Arguments

*   `--symbols SYMBOLS`: **Required**. A comma-separated list of trading symbols to fetch data for (e.g., `BTCUSDT,ETHUSDT,ADAUSDT`). Symbols should match Binance's naming.
*   `--intervals INTERVALS`: **Required**. A comma-separated list of k-line intervals (e.g., `1m,5m,15m,1h,4h,1d,1w`).
*   `--start_date START_DATE`: **Required**. The start date for data fetching. Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS".
*   `--end_date END_DATE`: **Optional**. The end date for data fetching. Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS". If not provided, data will be fetched up to the current time.

### Data Storage

*   Fetched data is stored in CSV files.
*   **Location**: `data/binance/{SYMBOL}/{INTERVAL}/`
*   **Filename Convention**: `{SYMBOL}_{INTERVAL}_{START_OPEN_TIME_MS}.csv` (where `START_OPEN_TIME_MS` is the millisecond timestamp of the first k-line in the file).
*   **Format**: Each row in the CSV represents a single k-line with the following columns: `open_time`, `open`, `high`, `low`, `close`, `volume`, `close_time`, `quote_asset_volume`, `number_of_trades`, `taker_buy_base_asset_volume`, `taker_buy_quote_asset_volume`.

## Disclaimer

Trading cryptocurrencies involves significant risk. This software is provided "as is" without warranty of any kind. The developers are not responsible for any financial losses incurred through the use of this software. Always do your own research and exercise caution.

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

## Setup & Installation

1.  **Freqtrade**: Ensure you have a working Freqtrade installation. Freqtrade is now installed as a Python package.
2.  **Python Environment**: Set up a Python environment (e.g., venv) with dependencies from `requirements.txt`.
    ```bash
    python -m venv .venv
    source .venv/bin/activate # or .venv\Scripts\activate for Windows
    pip install -r requirements.txt
    ```
3.  **API-sleutels (`.env`):**
    Creëer een bestand genaamd `.env` in de hoofdmap van het project. Vul dit bestand met je API-sleutels voor OpenAI, Grok en Bitvavo. **Dit bestand mag NOOIT naar GitHub worden gecommit.**
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    OPENAI_MODEL="gpt-4o"
    GROK_API_KEY="grok-YOUR_GROK_API_KEY"
    GROK_MODEL="grok-1"
    GROK_LIVE_SEARCH_API_URL="https://api.x.ai/v1/live-search" # Pas dit aan naar het officiële Grok Live Search API endpoint zodra bekend
    BITVAVO_API_KEY="YOUR_BITVAVO_API_KEY"
    BITVAVO_SECRET_KEY="YOUR_BITVAVO_SECRET_KEY"
    ```
4.  **Freqtrade Configuration (`config/config.json`):**
    Refer to the "Configuration" section below for details on setting up `config/config.json`. The base configuration is present in the repository.
5.  **Database**: Ensure Freqtrade is configured to use an SQLite database (e.g., `freqtrade.sqlite`), as this is used by the AI components for performance analysis.

## Configuration

### Freqtrade `config/config.json`

*   **Pair Whitelist**: The `pair_whitelist` in your Freqtrade `config.json` should be carefully selected. The AI system will operate on these pairs. The `pair_whitelist` has been updated with several commonly traded pairs. The currently included example pairs are: `"ETH/EUR", "BTC/EUR", "ZEN/EUR", "WETH/USDT", "USDC/USDT", "WBTC/USDT", "LINK/USDT", "UNI/USDT", "ZEN/BTC", "LSK/BTC", "ETH/BTC"`.
    **Important**: Controleer de beschikbaarheid van alle gewenste paren op uw exchange (e.g., Bitvavo), aangezien niet alle exchanges alle cross-paren ondersteunen (bijvoorbeeld `LSK/BTC`).
    ```json
    "exchange": {
        "pair_whitelist": [
            "ETH/EUR",
            "BTC/EUR",
            // ... other pairs ...
            "LSK/BTC",
            "ETH/BTC"
        ]
        // ... other exchange settings
    },
    ```
*   **API Keys in `config.json`**: The API keys (`key`, `secret`) specified within `config/config.json` for the exchange are typically placeholders. In a live Freqtrade setup, these are often overridden by environment variables (which can be loaded from the `.env` file described in the "Setup & Installation" section).
*   **Dynamic Adjustments**: The system's approach to dynamic Freqtrade configuration adjustments involves the AI providing advice or suggesting modifications to strategy parameters or filter lists. These are not "hot-swaps" of the main Freqtrade `config.json` during live operations but rather updates to strategy-specific parameters or through controlled mechanisms that Freqtrade supports for dynamic loading.

## Critical Testing Advisory

**IMPORTANT**: Before considering any live deployment, this system requires **extensive backtesting and prolonged dry-run testing** on a compatible Freqtrade setup. AI-driven trading strategies can be complex and may introduce unforeseen risks.

*   **Backtesting**: Use Freqtrade's backtesting capabilities to evaluate strategy performance over historical data.
*   **Dry-Run**: Run the bot in a simulated environment (dry-run mode) on a live market to observe its behavior without risking real capital.
*   **Gradual Exposure**: If moving to live trading, start with minimal capital and closely monitor performance and behavior.

## Usage (Conceptual)

The primary entry point for the AI optimization logic is `main.py`.

```bash
python main.py
```

This would typically initialize the `AIOptimizer` and start its periodic optimization cycles, interacting with a running Fregtrade instance or its data.

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

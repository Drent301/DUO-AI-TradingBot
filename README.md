# DUO-AI-TradingBot

Reflective AI-powered crypto trading bot using Freqtrade, Bitvavo, GPT, and Grok for autonomous learning.

## Overview

DUO-AI-TradingBot is an advanced trading bot that leverages artificial intelligence to make informed trading decisions. It integrates with Freqtrade and utilizes AI models like GPT and Grok for market analysis, reflection, and continuous self-improvement. The bot aims to learn from its trades and adapt its strategies over time.

## Key Features & Current Status

### Trading Pairs (`pair_whitelist`)

The bot is configured to trade a specific set of cryptocurrency pairs. The current `pair_whitelist` includes:
- ETH/BTC
- ETH/EUR
- LSK/BTC
- ZEN/BTC

This list is managed within the Freqtrade configuration and is crucial for defining the bot's trading scope.

### CNN Patterns (`core/cnn_patterns.py`)

The `cnn_patterns.py` module is responsible for detecting candlestick patterns.
-   **Current Implementation**: Currently, this module operates on a rule-based system to identify common patterns.
-   **Future Enhancement (Critical)**: A key future development is the integration of trained Machine Learning (ML) models into this module. This will allow for more nuanced and adaptive pattern detection, moving beyond predefined rules to learned pattern recognition, significantly enhancing the bot's analytical capabilities.

### Data Source for AI Analysis

-   **Exclusive Reliance on Freqtrade Database**: All data required for AI analysis, including historical trade data and market information, is now sourced exclusively from Freqtrade's internal database.
-   **`trade_logger.py` (Removed)**: The custom `core/trade_logger.py` module has been removed from the project. Its functionality for logging trades has been delegated to Freqtrade's robust internal data management, streamlining the data pipeline.

### Dynamic Freqtrade Configuration

The bot's AI modules can generate advice or filters that may suggest modifications to the Freqtrade configuration (e.g., adjusting strategy parameters).
-   **Current Approach**: Such advice is typically implemented as dynamic parameters within strategies or as inputs to decision-making logic.
-   **Important Note on Runtime Changes**: Direct runtime modifications to Freqtrade's core configuration files (e.g., `config.json`) that affect fundamental bot operations (like `pair_whitelist` changes or exchange settings) **require a full bot restart** to take effect. The AI learns and suggests, but applying certain structural changes needs manual intervention and a restart.

### AI Cooldowns (`core/cooldown_tracker.py`)

A new module, `core/cooldown_tracker.py`, has been introduced to manage AI-specific cooldown periods for trading pairs within specific strategies.
-   This system is complementary to Freqtrade's built-in cooldown mechanisms.
-   It allows the AI to enforce a temporary cessation of trading for a pair/strategy combination, for instance, after a significant loss or when market conditions are deemed too uncertain by the AI, based on the learnable `cooldownDurationSeconds` parameter managed by `ParamsManager`.

### `preferredPairs` Learning

The AI system now includes logic for learning `preferredPairs`. This involves analyzing market conditions and performance to identify pairs that are historically more favorable for trading under certain AI strategies.
-   **Current Implementation**: The learning logic for `preferredPairs` is implemented.
-   **Integration with `pair_whitelist`**: For the bot to act on these learned preferences by changing Freqtrade's active `pair_whitelist`, **a bot restart is currently required**. The AI can identify preferred pairs, but updating the live trading scope needs this manual step.

### Formal Test Suite (Missing)

-   **Critical Need**: The project currently lacks a formal, comprehensive test suite (e.g., using `pytest` or Python's `unittest`).
-   **Future Work**: Implementing a robust test suite is a high-priority task. This will be crucial for ensuring the reliability, correctness, and stability of all modules, especially the core AI logic, data handling, and Freqtrade interactions. Tests will cover unit, integration, and potentially end-to-end scenarios.

## Core AI Modules

The bot's intelligence is driven by several core Python modules located in the `core/` directory:
-   `params_manager.py`: Manages learnable parameters and hyperparameters for the AI.
-   `entry_decider.py`: Determines optimal entry points for trades.
-   `exit_optimizer.py`: Optimizes exit points and manages trailing stop losses.
-   `bias_reflector.py`: Assesses and adjusts for market bias.
-   `confidence_engine.py`: Calculates the AI's confidence in its decisions.
-   `reflectie_lus.py` & `reflectie_analyser.py`: Handle post-trade reflection and learning.
-   `strategy_manager.py`: Manages and potentially mutates trading strategies.
-   `pre_trainer.py`: (Details to be added - likely for pre-training ML models or initializing parameters).
-   `ai_optimizer.py`: (Details to be added - likely for optimizing AI model parameters or strategy configurations).
-   And more, interacting with GPT and Grok APIs.

## Setup & Running

(Details to be added regarding Freqtrade setup, API key configuration for Bitvavo, GPT, Grok, and how to run the bot.)

## Contributing

(Details to be added for contribution guidelines.)

## License

(Details to be added - e.g., MIT License.)

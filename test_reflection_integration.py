import asyncio
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
import logging # Added for logging within the test script

# Configure basic logging for the test script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure core modules are importable by adjusting Python path if necessary
# This might require specific setup if run from a different directory than the project root
import sys
# Assuming the script is run from the project root, or core is in PYTHONPATH
# If core is a top-level directory:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# If the script is in /app and core is /app/core:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))


from core.params_manager import ParamsManager
from core.reflectie_lus import ReflectieLus, MEMORY_DIR as REFLECTIE_LUS_MEMORY_DIR
# ReflectieAnalyser and other components are used internally by ReflectieLus

# Simplified DUOAI_Strategy for testing
class MockDUOAIStrategy:
    def __init__(self, config, params_manager_instance):
        self.config = config
        self.name = config.get('strategy_name', 'DUOAI_Strategy')
        self.params_manager = params_manager_instance

        self.learned_ema_period = 20 # Default
        self.learned_rsi_threshold_buy = 30 # Default
        self.learned_rsi_threshold_sell = 70 # Default

        self.minimal_roi = {"0": 0.1} # Freqtrade standard
        self.stoploss = -0.1 # Freqtrade standard
        self.trailing_stop = False
        self.trailing_stop_positive = None
        self.trailing_stop_positive_offset = 0.0
        self.trailing_only_offset_is_reached = False


        self.dp = MagicMock()
        self.wallets = MagicMock()
        self.proposal_file_path = os.path.join(REFLECTIE_LUS_MEMORY_DIR, f"latest_mutation_proposal_{self.name}.json")
        logging.info(f"MockDUOAIStrategy initialized for {self.name}. Proposal file: {self.proposal_file_path}")


    def _load_and_apply_learned_parameters(self):
        logging.info(f"Strategy {self.name}: Loading learned parameters...")
        # In a real strategy, this would be self.params_manager.get_param("strategies", strategy_id=self.name)
        # For the mock, we assume the strategy block is directly what we set in pm.params["strategies"][self.name]
        strategy_settings_block = self.params_manager._params.get("strategies", {}).get(self.name, {})


        if strategy_settings_block and isinstance(strategy_settings_block, dict):
            self.learned_ema_period = strategy_settings_block.get('learned_ema_period', self.learned_ema_period)
            self.learned_rsi_threshold_buy = strategy_settings_block.get('learned_rsi_threshold_buy', self.learned_rsi_threshold_buy)
            self.learned_rsi_threshold_sell = strategy_settings_block.get('learned_rsi_threshold_sell', self.learned_rsi_threshold_sell)

            self.minimal_roi = strategy_settings_block.get("minimal_roi", self.minimal_roi)
            self.stoploss = strategy_settings_block.get("stoploss", self.stoploss)
            logging.info(f"Loaded: EMA={self.learned_ema_period}, RSI Buy={self.learned_rsi_threshold_buy}, RSI Sell={self.learned_rsi_threshold_sell}")
        else:
            logging.info(f"No strategy-specific settings block found for {self.name} in params_manager. Using defaults.")

    async def _apply_mutation_proposals(self):
        logging.info(f"Strategy {self.name}: Attempting to apply mutation proposals from {self.proposal_file_path}...")
        if not os.path.exists(self.proposal_file_path):
            logging.info("No proposal file found.")
            return

        try:
            with open(self.proposal_file_path, 'r') as f:
                proposal = json.load(f)
        except Exception as e:
            logging.error(f"Error loading proposal file: {e}")
            if os.path.exists(self.proposal_file_path): os.remove(self.proposal_file_path)
            return

        threshold = self.params_manager.get_param("apply_mutation_threshold_confidence", default=0.65)

        if (proposal and proposal.get('strategyId') == self.name and
                proposal.get('confidence', 0) >= threshold):
            logging.info(f"Applying proposal with confidence {proposal['confidence']:.2f} (Threshold: {threshold:.2f})")
            changes = proposal.get('adjustments', {}).get('parameterChanges', {})
            applied_changes_dict = {}

            if 'emaPeriod' in changes:
                self.learned_ema_period = changes['emaPeriod']
                applied_changes_dict['learned_ema_period'] = self.learned_ema_period
                logging.info(f"Applied emaPeriod: {self.learned_ema_period}")
            if 'rsiThresholdBuy' in changes:
                self.learned_rsi_threshold_buy = changes['rsiThresholdBuy']
                applied_changes_dict['learned_rsi_threshold_buy'] = self.learned_rsi_threshold_buy
                logging.info(f"Applied rsiThresholdBuy: {self.learned_rsi_threshold_buy}")
            if 'rsiThresholdSell' in changes:
                self.learned_rsi_threshold_sell = changes['rsiThresholdSell']
                applied_changes_dict['learned_rsi_threshold_sell'] = self.learned_rsi_threshold_sell
                logging.info(f"Applied rsiThresholdSell: {self.learned_rsi_threshold_sell}")

            if applied_changes_dict:
                # This part simulates how DUOAI_Strategy would call set_param
                # For the mock, we'll directly update the in-memory params of our pm instance
                if "strategies" not in self.params_manager._params: self.params_manager._params["strategies"] = {}
                if self.name not in self.params_manager._params["strategies"]: self.params_manager._params["strategies"][self.name] = {}
                self.params_manager._params["strategies"][self.name].update(applied_changes_dict)
                logging.info(f"Updated parameters for {self.name} in mocked ParamsManager: {applied_changes_dict}")
        else:
            reason = "proposal invalid"
            if proposal and proposal.get('strategyId') != self.name : reason = "strategy ID mismatch"
            elif proposal and proposal.get('confidence',0) < threshold : reason = f"confidence too low ({proposal.get('confidence',0):.2f} < {threshold:.2f})"
            logging.info(f"Proposal not applied. Reason: {reason}.")

        if os.path.exists(self.proposal_file_path):
            os.remove(self.proposal_file_path)
            logging.info(f"Removed proposal file: {self.proposal_file_path}")

async def run_reflection_loop_test_main_function():
    logging.info("--- Starting Reflection Loop Integration Test ---")
    if not os.path.exists(REFLECTIE_LUS_MEMORY_DIR):
        os.makedirs(REFLECTIE_LUS_MEMORY_DIR)
        logging.info(f"Created MEMORY_DIR: {REFLECTIE_LUS_MEMORY_DIR}")


    pm = ParamsManager()
    initial_pm_params = {
        "global": { # Added global key
            "apply_mutation_threshold_confidence": 0.60,
            "freqtrade_db_url": "sqlite:///mock_trades.sqlite",
            "performance_lookback_days": 7,
            "predictAdjust_adjStrengthenThreshold": 70,
            "predictAdjust_paramStrengthenMultiplier": 1.1,
            "predictAdjust_avgProfitScaleFactor": 600,
            "predictAdjust_avgProfitScoreMinMax": 30,
            "predictAdjust_winRateWeight": 50,
            "predictAdjust_tradeCountDivisor": 5,
            "predictAdjust_tradeCountScoreMax": 20,
            "predictAdjust_biasHighThreshold": 0.7,
            "predictAdjust_biasLowThreshold": 0.3,
            "predictAdjust_biasInfluenceMultiplier": 30
        },
        "strategies": {
            "DUOAI_Strategy": {
                "learned_ema_period": 20,
                "learned_rsi_threshold_buy": 30,
                "learned_rsi_threshold_sell": 70,
                "minimal_roi": {"0": 0.05},
                "stoploss": -0.10
            }
        }
    }
    # Directly set the ._params attribute for this test instance
    pm._params = initial_pm_params
    logging.info(f"ParamsManager initialized with test params: {pm._params}")


    strategy_config = {'strategy_name': 'DUOAI_Strategy', 'timeframe': '5m'}
    strategy = MockDUOAIStrategy(strategy_config, pm)

    reflectie_lus_instance = ReflectieLus(params_manager=pm)

    mock_gpt_response = {"reflectie": "GPT says buy.", "confidence": 0.8, "bias": 0.7, "intentie": "bullish", "emotie": "excited"}
    mock_grok_response = {"reflectie": "Grok agrees, strong buy.", "confidence": 0.9, "bias": 0.75, "intentie": "very bullish", "emotie": "optimistic"}
    mock_cnn_patterns = {
        "overall_summary": "Market looks bullish overall.",
        "key_observations": ["Strong support at previous lows."],
        "emerging_patterns": [{"pattern_name":"bullFlag", "timeframe":"1h", "strength":0.8}],
        "contextual_info": {"trend": "uptrend", "volume_spike": True, "volatility":"medium"},
        "timeframe_details": {
            "5m": {"patterns": ["doji"], "prediction_label": "bullish", "prediction_confidence":0.6, "market_condition":"trending"},
            "1h": {"patterns": ["bullishEngulfing"], "prediction_label": "strong_bullish", "prediction_confidence":0.85, "market_condition":"strong_uptrend"}
        }
    }
    mock_sentiment_data = [{"title": "ETH to the moon!", "source": "twitter", "content": "Big news coming for ETH.", "sentiment_score":0.8, "sentiment_label":"positive"}]
    strong_mock_performance = {"winRate": 0.8, "avgProfit": 0.05, "tradeCount": 100, "totalProfit": 5.0, "sharpeRatio": 2.1, "maxDrawdown": 0.05}


    with patch.object(reflectie_lus_instance.gpt_reflector, 'ask_ai', AsyncMock(return_value=mock_gpt_response)), \
         patch.object(reflectie_lus_instance.grok_reflector, 'ask_grok', AsyncMock(return_value=mock_grok_response)), \
         patch.object(reflectie_lus_instance.prompt_builder.cnn_patterns, 'detect_patterns_multi_timeframe', AsyncMock(return_value=mock_cnn_patterns)), \
         patch.object(reflectie_lus_instance.prompt_builder.sentiment_fetcher if reflectie_lus_instance.prompt_builder.sentiment_fetcher else MagicMock(), 'fetch_live_search_data', AsyncMock(return_value=mock_sentiment_data)), \
         patch.object(reflectie_lus_instance.performance_monitor, 'get_strategy_performance', AsyncMock(return_value=strong_mock_performance)):

        logging.info("--- Cycle 1: Initial State & First Reflection ---")
        strategy._load_and_apply_learned_parameters()
        assert strategy.learned_ema_period == 20, f"Initial EMA period mismatch: Expected 20, Got {strategy.learned_ema_period}"

        mock_candles_df = pd.DataFrame({
            'close': [10,11,12,13,14], 'open': [9,10,11,12,13],
            'high': [11,12,13,14,15], 'low': [9,10,11,12,13], 'volume': [100,110,120,130,140]
        }, index=pd.to_datetime([datetime.now() - timedelta(minutes=i*5) for i in range(5)]))
        mock_candles_by_timeframe = {"5m": mock_candles_df, "1h": mock_candles_df.copy()}
        mock_trade_context = {"profit_pct": 0.02, "exit_reason": "roi", "pair": "ETH/EUR"}

        await reflectie_lus_instance.process_reflection_cycle(
            symbol="ETH/EUR",
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=strategy.name,
            trade_context=mock_trade_context,
            current_bias=0.7,
            current_confidence=0.8,
            mode='live'
        )

        assert os.path.exists(strategy.proposal_file_path), "Proposal file not created"
        with open(strategy.proposal_file_path, 'r') as f:
            proposal_content = json.load(f)
        logging.info(f"Generated proposal: {proposal_content}")

        # Check based on default params for ReflectieAnalyser.predict_strategy_adjustment
        # Default: adjStrengthenThreshold=70. Performance score should be high with winRate=0.8, avgProfit=0.05
        # Perf score = (0.8*50) + min(max(0.05*600,-30),30) + min(100/5,20) = 40 + 30 + 20 = 90
        # Bias 0.7. adj_score = 90. 90 > 70, so "strengthen".
        assert proposal_content['adjustments']['action'] == 'strengthen', \
            f"Expected 'strengthen', got {proposal_content['adjustments']['action']}. Score: {proposal_content.get('currentAdjustmentScore')}"

        await strategy._apply_mutation_proposals() # Strategy applies the proposal

        # Default strengthen multiplier is 1.1
        expected_ema = int(20 * 1.1)
        assert strategy.learned_ema_period == expected_ema, \
            f"EMA period not updated as expected: Expected {expected_ema}, Got {strategy.learned_ema_period}"

        final_strat_params_block = pm._params["strategies"][strategy.name]
        assert final_strat_params_block['learned_ema_period'] == expected_ema, \
            "EMA period not persisted in ParamsManager as expected"

        logging.info("--- Cycle 1 Test Passed: Parameters mutated and persisted. ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_reflection_loop_test_main_function())
    finally:
        test_proposal_file = os.path.join(REFLECTIE_LUS_MEMORY_DIR, f"latest_mutation_proposal_DUOAI_Strategy.json")
        if os.path.exists(test_proposal_file):
            os.remove(test_proposal_file)
            logging.info(f"Cleaned up test proposal file: {test_proposal_file}")

        # Clean up REFLECTIE_LOG, ANALYSE_LOG, BIAS_OUTCOME_LOG
        for log_file_path in [
            os.path.join(REFLECTIE_LUS_MEMORY_DIR, 'reflectie-logboek.json'),
            os.path.join(REFLECTIE_LUS_MEMORY_DIR, 'reflectie-analyse.json'),
            os.path.join(REFLECTIE_LUS_MEMORY_DIR, 'bias-outcome-log.json')
        ]:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
                logging.info(f"Cleaned up log file: {log_file_path}")

        # Attempt to remove MEMORY_DIR if empty
        if os.path.exists(REFLECTIE_LUS_MEMORY_DIR) and not os.listdir(REFLECTIE_LUS_MEMORY_DIR):
            try:
                os.rmdir(REFLECTIE_LUS_MEMORY_DIR)
                logging.info(f"Cleaned up empty MEMORY_DIR: {REFLECTIE_LUS_MEMORY_DIR}")
            except OSError as e:
                logging.warning(f"Could not remove MEMORY_DIR {REFLECTIE_LUS_MEMORY_DIR}: {e}")

        logging.info("--- Reflection Loop Integration Test Finished ---")

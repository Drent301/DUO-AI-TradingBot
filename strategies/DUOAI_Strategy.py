# strategies/DUOAI_Strategy.py
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np
import os
import json
from functools import reduce # Import reduce

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade

from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.cnn_patterns import CNNPatterns
from core.reflectie_lus import ReflectieLus, MEMORY_DIR as REFLECTIE_LUS_MEMORY_DIR
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.entry_decider import EntryDecider
from core.exit_optimizer import ExitOptimizer
from core.strategy_manager import StrategyManager
from core.interval_selector import IntervalSelector
from core.params_manager import ParamsManager
from core.trade_logger import TradeLogger
from core.cooldown_tracker import CooldownTracker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DUOAI_Strategy(IStrategy):
    minimal_roi = {"0": 0.05, "30": 0.03, "60": 0.02, "120": 0.01}
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True
    timeframe = '5m'
    startup_candle_count = 100

    CONCEPTUAL_PAIR_WHITELIST = [
        "ETH/EUR", "ETH/BTC", "WETH/USDT", "BTC/EUR", "ZEN/EUR", "ZEN/BTC",
        "USDC/USDT", "WBTC/USDT", "LINK/USDT", "UNI/USDT", "LSK/BTC"
    ]
    informative_timeframes = ['1h', '4h', '1d']
    informative_pairs = [(pair, tf) for pair in CONCEPTUAL_PAIR_WHITELIST for tf in informative_timeframes]

    # Instantiate AI modules
    # ReflectieLus needs ParamsManager.
    # We create a single instance of ParamsManager first, then pass it around.
    params_manager_instance = ParamsManager() # Central instance

    prompt_builder: PromptBuilder = PromptBuilder()
    gpt_reflector: GPTReflector = GPTReflector()
    grok_reflector: GrokReflector = GrokReflector()
    cnn_patterns: CNNPatterns = CNNPatterns()
    reflectie_lus: ReflectieLus = ReflectieLus(params_manager=params_manager_instance)
    bias_reflector: BiasReflector = BiasReflector()
    confidence_engine: ConfidenceEngine = ConfidenceEngine()
    entry_decider: EntryDecider = EntryDecider()
    exit_optimizer: ExitOptimizer = ExitOptimizer()
    strategy_manager: StrategyManager = StrategyManager()
    interval_selector: IntervalSelector = IntervalSelector()
    params_manager: ParamsManager = params_manager_instance # Use the same instance
    trade_logger: TradeLogger = TradeLogger()
    cooldown_tracker: CooldownTracker = CooldownTracker()

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.learned_ema_period = 20
        self.learned_rsi_threshold_buy = 30
        self.learned_rsi_threshold_sell = 70
        # ReflectieLus is already instantiated at class level with params_manager
        # If ReflectieLus needed specific config from strategy `config` dict,
        # it would need to be re-initialized or updated here.

        initial_pair_for_params = config.get('pair_whitelist', [''])[0] if config.get('pair_whitelist') else 'default'
        self._load_and_apply_learned_parameters(initial_pair_for_params)
        logger.info(f"DUOAI_Strategy geïnitialiseerd. Learned EMA: {self.learned_ema_period}, RSI Buy: {self.learned_rsi_threshold_buy}, RSI Sell: {self.learned_rsi_threshold_sell}")

    def _get_all_relevant_candles_for_ai(self, pair: str) -> Dict[str, pd.DataFrame]:
        candles_by_timeframe: Dict[str, pd.DataFrame] = {}
        base_df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if not base_df.empty:
            candles_by_timeframe[self.timeframe] = base_df.copy()
        else:
            logger.warning(f"Geen basis dataframe voor {pair} op {self.timeframe}.")
            return {}
        for tf in self.informative_timeframes:
            if tf == self.timeframe: continue
            try:
                informative_df = self.dp.get_pair_dataframe(pair, tf)
                if not informative_df.empty:
                    candles_by_timeframe[tf] = informative_df.copy()
                else:
                    logger.debug(f"Informative dataframe for {pair} on {tf} is empty.")
            except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen: {e}")
        return candles_by_timeframe

    def _load_and_apply_learned_parameters(self, pair: str) -> None:
        strategy_params = self.params_manager.get_param("strategies", strategy_id=self.name)
        if strategy_params and isinstance(strategy_params, dict):
            self.minimal_roi = strategy_params.get("minimal_roi", self.minimal_roi)
            self.stoploss = strategy_params.get("stoploss", self.stoploss)
            self.trailing_stop = strategy_params.get("trailing_stop", self.trailing_stop)
            self.trailing_stop_positive = strategy_params.get("trailing_stop_positive", self.trailing_stop_positive)
            self.trailing_stop_positive_offset = strategy_params.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_only_offset_is_reached = strategy_params.get("trailing_only_offset_is_reached", self.trailing_only_offset_is_reached)

            self.learned_ema_period = strategy_params.get("learned_ema_period", self.learned_ema_period)
            self.learned_rsi_threshold_buy = strategy_params.get("learned_rsi_threshold_buy", self.learned_rsi_threshold_buy)
            self.learned_rsi_threshold_sell = strategy_params.get("learned_rsi_threshold_sell", self.learned_rsi_threshold_sell)
            logger.info(f"Strategie parameters voor {self.name} (pair: {pair}) geladen: EMA={self.learned_ema_period}, RSI Buy={self.learned_rsi_threshold_buy}, RSI Sell={self.learned_rsi_threshold_sell}")
        else:
            logger.warning(f"Geen strategie-specifieke parameters voor {self.name} in params_manager. Gebruikt init defaults.")

    def _apply_mutation_proposals(self) -> None:
        proposal_file_path = os.path.join(REFLECTIE_LUS_MEMORY_DIR, f"latest_mutation_proposal_{self.name}.json")
        if not os.path.exists(proposal_file_path):
            logger.debug(f"Geen mutatievoorstel bestand op {proposal_file_path}")
            return
        try:
            with open(proposal_file_path, 'r', encoding='utf-8') as f:
                proposal = json.load(f)
        except Exception as e:
            logger.error(f"Fout bij laden/decoderen mutatievoorstel {proposal_file_path}: {e}")
            try: os.remove(proposal_file_path)
            except OSError: pass
            return

        if not isinstance(proposal, dict) or proposal.get('strategyId') != self.name or \
           'confidence' not in proposal or 'adjustments' not in proposal or \
           not isinstance(proposal['adjustments'], dict):
            logger.warning(f"Ongeldig mutatievoorstel van {proposal_file_path}: {proposal}")
            try: os.remove(proposal_file_path)
            except OSError: pass
            return

        apply_threshold = self.params_manager.get_param("apply_mutation_threshold_confidence", default=0.65)
        if proposal['confidence'] >= apply_threshold:
            logger.info(f"Toepassen mutatievoorstel ({proposal['adjustments'].get('action', 'N/A')}) voor {self.name} met confidence {proposal['confidence']:.2f}")
            parameter_changes = proposal['adjustments'].get('parameterChanges', {})
            changed_log = []
            params_to_persist = {}

            if 'emaPeriod' in parameter_changes and self.learned_ema_period != int(parameter_changes['emaPeriod']):
                self.learned_ema_period = int(parameter_changes['emaPeriod'])
                changed_log.append(f"emaPeriod -> {self.learned_ema_period}")
                params_to_persist["learned_ema_period"] = self.learned_ema_period
            if 'rsiThresholdBuy' in parameter_changes and self.learned_rsi_threshold_buy != int(parameter_changes['rsiThresholdBuy']):
                self.learned_rsi_threshold_buy = int(parameter_changes['rsiThresholdBuy'])
                changed_log.append(f"rsiThresholdBuy -> {self.learned_rsi_threshold_buy}")
                params_to_persist["learned_rsi_threshold_buy"] = self.learned_rsi_threshold_buy
            if 'rsiThresholdSell' in parameter_changes and self.learned_rsi_threshold_sell != int(parameter_changes['rsiThresholdSell']):
                self.learned_rsi_threshold_sell = int(parameter_changes['rsiThresholdSell'])
                changed_log.append(f"rsiThresholdSell -> {self.learned_rsi_threshold_sell}")
                params_to_persist["learned_rsi_threshold_sell"] = self.learned_rsi_threshold_sell

            if changed_log:
                logger.info(f"Toegepaste parameterwijzigingen voor {self.name}: {', '.join(changed_log)}")
                for key, val in params_to_persist.items():
                    asyncio.create_task(self.params_manager.set_param(key, val, strategy_id=self.name))
            else:
                logger.info(f"Mutatievoorstel bevatte geen nieuwe parameterwaarden voor {self.name}.")
        else:
            logger.info(f"Mutatievoorstel niet toegepast, confidence {proposal['confidence']:.2f} < drempel {apply_threshold:.2f}.")

        try: os.remove(proposal_file_path)
        except OSError as e: logger.error(f"Fout bij verwijderen mutatievoorstel {proposal_file_path}: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        self._load_and_apply_learned_parameters(metadata['pair'])
        self._apply_mutation_proposals()

        dataframe['ema_learned'] = ta.EMA(dataframe, timeperiod=self.learned_ema_period)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        logger.info(f"Indicators voor {metadata['pair']}: EMA({self.learned_ema_period})={dataframe['ema_learned'].iloc[-1]:.2f}. "
                    f"RSI(14)={dataframe['rsi'].iloc[-1]:.2f}. "
                    f"Te gebruiken RSI thresholds (Buy: {self.learned_rsi_threshold_buy}, Sell: {self.learned_rsi_threshold_sell}).")

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(20).mean()

        for pair_to_merge, informative_tf_to_merge in self.informative_pairs:
            if pair_to_merge == metadata['pair'] and informative_tf_to_merge != self.timeframe:
                informative_df = self.dp.get_pair_dataframe(pair_to_merge, informative_tf_to_merge)
                if not informative_df.empty:
                    dataframe = merge_informative_pair(dataframe, informative_df, self.timeframe, informative_tf_to_merge, fqtrade=True)
                else:
                    logger.debug(f"Informative dataframe for {pair_to_merge}/{informative_tf_to_merge} is empty.")

        current_bias = self.bias_reflector.get_bias_score(metadata['pair'], self.name)
        current_confidence = self.confidence_engine.get_confidence_score(metadata['pair'], self.name)
        dataframe['ai_bias'] = current_bias
        dataframe['ai_confidence'] = current_confidence
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            leverage: float, default_stake_amount: float, **kwargs) -> float:
        max_per_trade_pct_learned = self.confidence_engine.confidence_memory.get(pair, {}).get(self.name, {}).get('max_per_trade_pct', 0.1)
        global_max_trade_risk = self.params_manager.get_param("maxTradeRiskPct", strategy_id=None)
        effective_max_trade_risk = min(max_per_trade_pct_learned, global_max_trade_risk)
        total_capital = self.wallets.get_leverage_capital(self.stake_currency)
        adjusted_stake_amount = total_capital * effective_max_trade_risk
        if self.stake_amount != "unlimited":
            adjusted_stake_amount = min(adjusted_stake_amount, float(self.stake_amount))
        min_stake_from_config = self.config.get('exchange', {}).get('min_stake_amount', 10.0)
        adjusted_stake_amount = max(adjusted_stake_amount, min_stake_from_config)
        logger.info(f"Aangepaste stake amount voor {pair}: {adjusted_stake_amount:.2f} {self.stake_currency} (MaxPerTradePct: {effective_max_trade_risk:.2%}).")
        return adjusted_stake_amount

    async def custom_entry(self, pair: str, current_time: datetime,
                           dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai:
            logger.warning(f"Geen dataframes voor AI-entrybesluit {pair}.")
            return None
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)
        if self.cooldown_tracker.is_cooldown_active(pair, self.name):
            cooldown_info = self.cooldown_tracker._cooldown_state.get(pair, {}).get(self.name, {})
            logger.info(f"Entry geweigerd {pair} door AI cooldown (reden: {cooldown_info.get('reason')}, eindigt: {cooldown_info.get('end_time')}).")
            return None
        entry_decision = await self.entry_decider.should_enter(
            dataframe=dataframe, symbol=pair, current_strategy_id=self.name,
            trade_context={"current_price": dataframe['close'].iloc[-1], "timeframe": self.timeframe, "candles_by_timeframe": candles_by_timeframe_for_ai},
            learned_bias=learned_bias, learned_confidence=learned_confidence,
            entry_conviction_threshold=entry_conviction_threshold
        )
        if entry_decision['enter']:
            logger.info(f"Entry toegestaan {pair}. Reden: {entry_decision['reason']}. Confidence: {entry_decision['confidence']:.2f}")
            return 1.0
        logger.info(f"Entry geweigerd {pair}. Reden: {entry_decision['reason']}")
        return None

    async def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                          dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai:
            logger.warning(f"Geen dataframes voor AI-exitbesluit {pair}. Exit overgeslagen.")
            return None
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        exit_conviction_drop_trigger = self.params_manager.get_param("exitConvictionDropTrigger", strategy_id=self.name)
        exit_decision = await self.exit_optimizer.should_exit(
            dataframe=dataframe, trade=trade.to_json(), symbol=pair, current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai, learned_bias=learned_bias,
            learned_confidence=learned_confidence, exit_conviction_drop_trigger=exit_conviction_drop_trigger
        )
        if exit_decision['exit']:
            logger.info(f"Exit getriggerd {pair}. Reden: {exit_decision['reason']}. Confidence: {exit_decision['confidence']:.2f}")
            await self.cooldown_tracker.activate_cooldown(pair, self.name, reason=f"ai_exit_{exit_decision['reason']}")
            return dataframe['close'].iloc[-1]
        sl_optimization_result = await self.exit_optimizer.optimize_trailing_stop_loss(
            dataframe=dataframe, trade=trade.to_json(), symbol=pair, current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai, learned_bias=learned_bias, learned_confidence=learned_confidence
        )
        if sl_optimization_result:
            self.stoploss = sl_optimization_result.get("stoploss", self.stoploss)
            self.trailing_stop_positive_offset = sl_optimization_result.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_stop_positive = sl_optimization_result.get("trailing_stop_positive", self.trailing_stop_positive)
            asyncio.create_task(self.params_manager.update_strategy_roi_sl_params(
                strategy_id=self.name, new_roi=self.minimal_roi, new_stoploss=self.stoploss,
                new_trailing_stop_positive=self.trailing_stop_positive,
                new_trailing_stop_positive_offset=self.trailing_stop_positive_offset
            ))
            logger.info(f"TSL parameters bijgewerkt {pair}: SL={self.stoploss}, TSL_Offset={self.trailing_stop_positive_offset}, TSL_Trigger={self.trailing_stop_positive}")
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        current_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        current_bias = self.bias_reflector.get_bias_score(pair, self.name)
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)
        if current_confidence < entry_conviction_threshold or current_bias < 0.5:
            logger.warning(f"Trade entry {pair} geweigerd in confirm. Conf ({current_confidence:.2f}) < drempel ({entry_conviction_threshold:.2f}) of Bias ({current_bias:.2f}) < 0.5.")
            return False
        logger.info(f"Trade entry {pair} bevestigd in confirm. Conf: {current_confidence:.2f}, Bias: {current_bias:.2f}.")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        logger.info(f"Trade exit {pair} bevestigd in confirm_trade_exit.")
        return True

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['enter_long'] = 0
        conditions = []
        conditions.append(dataframe['rsi'] < self.learned_rsi_threshold_buy)
        conditions.append(dataframe['ema_learned'] > dataframe['bb_middleband']) # Example other condition
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        conditions = []
        conditions.append(dataframe['rsi'] > self.learned_rsi_threshold_sell)
        # Example: conditions.append(dataframe['close'] < dataframe['bb_lowerband']) # Example other condition
        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'exit_long'] = 1
        return dataframe

    def process_stopped_trade(self, pair: str, trade: Trade, order: Dict[str, Any], **kwargs) -> None:
        profit_loss_pct = trade.calc_profit_ratio()
        trade_data = trade.to_json()
        trade_data['profit_pct'] = profit_loss_pct
        trade_data['exit_rate'] = order.get('price')
        trade_data['exit_type'] = order.get('ft_pair_exit_reason', 'unknown')
        logger.info(f"Trade gesloten {pair} (ID: {trade.id}). Resultaat: {profit_loss_pct:.2%}. Exit: {trade_data['exit_type']}. Trigger AI reflectie.")
        asyncio.create_task(self.trade_logger.log_trade(trade_data))
        if profit_loss_pct < 0:
            asyncio.create_task(self.cooldown_tracker.activate_cooldown(pair, self.name, reason=f"loss_{trade_data['exit_type']}_{profit_loss_pct:.2%}"))

        candles_by_timeframe_for_reflect = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_reflect:
            logger.warning(f"Geen dataframes voor post-trade reflectie {pair}. Reflectie overgeslagen.")
        else:
            # ReflectieLus instance is at class level, ensure it has params_manager if needed by its methods or sub-objects
            # If ReflectieLus was changed to take params_manager in __init__, this class-level instance needs update
            # Corrected: ReflectieLus is initialized with ParamsManager at class level now.
            asyncio.create_task(
                self.reflectie_lus.process_reflection_cycle(
                    symbol=pair, candles_by_timeframe=candles_by_timeframe_for_reflect, strategy_id=self.name,
                    trade_context=trade_data, current_bias=self.bias_reflector.get_bias_score(pair, self.name),
                    current_confidence=self.confidence_engine.get_confidence_score(pair, self.name),
                    mode=self.config.get('runmode', 'live')
                )
            )
        current_perf = self.strategy_manager.get_strategy_performance(self.name)
        new_trade_count = current_perf.get('tradeCount', 0) + 1
        old_total_profit = current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0)
        new_total_profit_value = old_total_profit + profit_loss_pct
        new_win_rate = (current_perf.get('winRate', 0.0) * current_perf.get('tradeCount', 0) + (1 if profit_loss_pct > 0 else 0)) / new_trade_count if new_trade_count > 0 else 0.0
        new_avg_profit = new_total_profit_value / new_trade_count if new_trade_count > 0 else 0.0
        asyncio.create_task(self.strategy_manager.update_strategy_performance(
            strategy_id=self.name,
            new_performance={"winRate": new_win_rate, "avgProfit": new_avg_profit, "tradeCount": new_trade_count}
        ))

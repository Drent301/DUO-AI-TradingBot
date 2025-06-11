# strategies/DUOAI_Strategy.py
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy.interface import IStrategy
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import Exchange
from freqtrade.persistence import Trade

# Importeer je eigen AI-modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
from core.cnn_patterns import CNNPatterns
from core.reflectie_lus import ReflectieLus
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine
from core.entry_decider import EntryDecider
from core.exit_optimizer import ExitOptimizer
from core.strategy_manager import StrategyManager
from core.interval_selector import IntervalSelector
from core.params_manager import ParamsManager # Import de ParamsManager
from core.trade_logger import TradeLogger # Import de TradeLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DUOAI_Strategy(IStrategy):
    """
    De hoofdstrategieklasse voor de DUO-AI Trading Bot.
    Deze strategie combineert Freqtrade met onze eigen AI-modules voor
    zelflerende besluitvorming, reflectie en optimalisatie.
    """

    # Freqtrade Hyperopt/Strategy parameters (initial defaults)
    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.02,
        "120": 0.01
    }
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    timeframe = '5m'
    startup_candle_count = 100

    # Define informative timeframes to fetch data for AI modules
    # These are required by `cnn_patterns` and `prompt_builder` for multi-timeframe context.
    informative_timeframes = ['1h', '4h', '1d'] # Consistent with AI module requirements

    # Maak instanties van je AI-modules
    prompt_builder: PromptBuilder = PromptBuilder()
    gpt_reflector: GPTReflector = GPTReflector()
    grok_reflector: GrokReflector = GrokReflector()
    cnn_patterns: CNNPatterns = CNNPatterns()
    reflectie_lus: ReflectieLus = ReflectieLus()
    bias_reflector: BiasReflector = BiasReflector()
    confidence_engine: ConfidenceEngine = ConfidenceEngine()
    entry_decider: EntryDecider = EntryDecider()
    exit_optimizer: ExitOptimizer = ExitOptimizer()
    strategy_manager: StrategyManager = StrategyManager()
    interval_selector: IntervalSelector = IntervalSelector()
    params_manager: ParamsManager = ParamsManager()
    trade_logger: TradeLogger = TradeLogger()


    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # We laden geleerde parameters in `populate_indicators` of in een custom method
        # want `dp` is hier nog niet beschikbaar.
        # De `_load_and_apply_learned_parameters` wordt nu periodiek aangeroepen.
        logger.info(f"DUOAI_Strategy geïnitialiseerd.")

    def _get_all_relevant_candles_for_ai(self, pair: str) -> Dict[str, pd.DataFrame]: # Corrected type hint for Dict value
        """
        Haalt alle relevante candles (basistimeframe + informatives) op voor AI-modules.
        """
        candles_by_timeframe: Dict[str, pd.DataFrame] = {} # Corrected type hint for Dict value

        # Huidige timeframe dataframe
        # In Freqtrade methods zoals `populate_indicators`, `custom_entry`, `custom_exit`,
        # de `dataframe` parameter is al het dataframe voor de basis timeframe.
        # Maar als deze method buiten die context (bijv. in een periodic task) worden aangeroepen,
        # moeten we `self.dp.get_pair_dataframe` gebruiken.
        # Voor de veiligheid, fetchen we hier altijd via `self.dp`.

        # Base timeframe
        base_df = self.dp.get_pair_dataframe(pair, self.timeframe)
        if not base_df.empty:
            candles_by_timeframe[self.timeframe] = base_df.copy()
        else:
            logger.warning(f"Geen basis dataframe voor {pair} op {self.timeframe}. AI-modules krijgen mogelijk incomplete data.")
            return {} # Als base DF leeg is, kunnen we niet verder.

        # Informative timeframes
        for tf in self.informative_timeframes:
            try:
                informative_df = self.dp.get_pair_dataframe(pair, tf)
                if not informative_df.empty:
                    candles_by_timeframe[tf] = informative_df.copy()
                else:
                    logger.debug(f"Informative dataframe for {pair} on {tf} is empty. Skipping.")
            except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen via self.dp: {e}")

        return candles_by_timeframe

    def _load_and_apply_learned_parameters(self, pair: str) -> None:
        """
        Laadt de laatst bekende geleerde parameters van de `params_manager`
        en past ze toe op de Freqtrade strategie parameters.
        Deze methode kan ook worden aangeroepen om periodiek te updaten.
        """
        strategy_params = self.params_manager.get_param("strategies", strategy_id=self.name)

        if strategy_params and isinstance(strategy_params, dict):
            # Update ROI, Stoploss, Trailing Stop parameters
            self.minimal_roi = strategy_params.get("minimal_roi", self.minimal_roi)
            self.stoploss = strategy_params.get("stoploss", self.stoploss)
            self.trailing_stop_positive = strategy_params.get("trailing_stop_positive", self.trailing_stop_positive)
            self.trailing_stop_positive_offset = strategy_params.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)

            logger.debug(f"Strategie parameters voor {pair} bijgewerkt door geleerde parameters: {strategy_params}")
        else:
            logger.debug(f"Geen geleerde parameters gevonden voor {self.name}. Gebruik standaardstrategieparameters.")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame: # Corrected type hint for dataframe
        """
        Voeg indicatoren toe aan de dataframe.
        """
        # Load and apply learned parameters at the start of populate_indicators
        # This will ensure the strategy always uses the latest learned parameters.
        self._load_and_apply_learned_parameters(metadata['pair'])

        # Standaard Freqtrade/TA-Lib indicatoren
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(20).mean()

        # Haal bias en confidence op (van AI-modules) en voeg toe aan dataframe
        current_bias = self.bias_reflector.get_bias_score(metadata['pair'], self.name)
        current_confidence = self.confidence_engine.get_confidence_score(metadata['pair'], self.name)
        dataframe['ai_bias'] = current_bias
        dataframe['ai_confidence'] = current_confidence

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            leverage: float, default_stake_amount: float, **kwargs) -> float:
        """
        Pas de inzet per trade aan op basis van AI-confidence (maxTradeRiskPct).
        """
        max_per_trade_pct_learned = self.confidence_engine.confidence_memory.get(pair, {}).get(self.name, {}).get('max_per_trade_pct', 0.1)
        global_max_trade_risk = self.params_manager.get_param("maxTradeRiskPct", strategy_id=None) # Global parameter

        effective_max_trade_risk = min(max_per_trade_pct_learned, global_max_trade_risk)

        total_capital = self.wallets.get_leverage_capital(self.stake_currency)

        adjusted_stake_amount = total_capital * effective_max_trade_risk

        if self.stake_amount != "unlimited":
            adjusted_stake_amount = min(adjusted_stake_amount, self.stake_amount)

        min_stake = 10.0 # Definieer een minimale inzet om fouten te voorkomen
        adjusted_stake_amount = max(adjusted_stake_amount, min_stake)

        logger.info(f"Aangepaste stake amount voor {pair}: {adjusted_stake_amount:.2f} {self.stake_currency} (gebaseerd op effectieve MaxPerTradePct: {effective_max_trade_risk:.2%}).")
        return adjusted_stake_amount

    async def custom_entry(self, pair: str, current_time: datetime,
                           dataframe: pd.DataFrame, **kwargs) -> Optional[float]: # Corrected type hint for dataframe
        """
        AI-gestuurde entry-beslissing.
        """
        # Haal alle relevante candles op voor AI-analyse
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai:
            logger.warning(f"Geen voldoende dataframes voor AI-entrybesluit voor {pair}. Entry geweigerd.")
            return None

        # Haal de geleerde bias, confidence, en entry threshold op
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)
        cooldown_duration_seconds = self.params_manager.get_param("cooldownDurationSeconds", strategy_id=None)

        # AI-specifieke cooldown check (als deze nog niet is geimplementeerd in entry_decider)
        # Dit is de `cooldownDuration` uit de manifesten.
        # Placeholder: Check hier of een AI-cooldown actief is (bijv. in bias_reflector opgeslagen)
        # of als laatste_verlies_tijd < cooldown_duration_seconds.
        # Hier kan een externe module voor cooldown tracking worden gebruikt.
        # Bijvoorbeeld: `if self.cooldown_tracker.is_cooldown_active(pair, self.name): return None`

        # Voor nu: Simpele cooldown check gebaseerd op laatste trade als voorbeeld
        # Dit vereist toegang tot trade history, die je kunt krijgen via `trade_logger.get_all_trades()`
        # of Freqtrade's database.
        # if await self._is_ai_cooldown_active(pair, cooldown_duration_seconds):
        #     logger.info(f"AI-specifieke cooldown actief voor {pair}. Entry geweigerd.")
        #     return None


        entry_decision = await self.entry_decider.should_enter(
            dataframe=dataframe, # Basis timeframe DF met alle indicatoren en gemergde informatives
            symbol=pair,
            current_strategy_id=self.name,
            trade_context={"current_price": dataframe['close'].iloc[-1], "timeframe": self.timeframe, "candles_by_timeframe": candles_by_timeframe_for_ai},
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            entry_conviction_threshold=entry_conviction_threshold
        )

        if entry_decision['enter']:
            logger.info(f"Entry toegestaan voor {pair}. Reden: {entry_decision['reason']}. Confidence: {entry_decision['confidence']:.2f}")
            return 1.0 # Return a value > 0 to indicate that an entry is desired.

        logger.info(f"Entry geweigerd voor {pair}. Reden: {entry_decision['reason']}")
        return None

    async def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                          dataframe: pd.DataFrame, **kwargs) -> Optional[float]: # Corrected type hint for dataframe
        """
        AI-gestuurde exit-beslissing en dynamische SL/TP aanpassing.
        """
        # Haal alle relevante candles op voor AI-analyse
        candles_by_timeframe_for_ai = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_ai:
            logger.warning(f"Geen voldoende dataframes voor AI-exitbesluit voor {pair}. Exit overgeslagen.")
            return None

        # Haal de geleerde bias en confidence op
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)

        # Haal de exit conviction drop trigger op van params_manager
        exit_conviction_drop_trigger = self.params_manager.get_param("exitConvictionDropTrigger", strategy_id=self.name)

        exit_decision = await self.exit_optimizer.should_exit(
            dataframe=dataframe,
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai,
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            exit_conviction_drop_trigger=exit_conviction_drop_trigger
        )

        if exit_decision['exit']:
            logger.info(f"Exit getriggerd voor {pair}. Reden: {exit_decision['reason']}. Confidence: {exit_decision['confidence']:.2f}")
            return dataframe['close'].iloc[-1]

        # Dynamische Trailing Stop Loss Optimalisatie (via AI)
        sl_optimization_result = await self.exit_optimizer.optimize_trailing_stop_loss(
            dataframe=dataframe,
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai,
            learned_bias=learned_bias,
            learned_confidence=learned_confidence
        )

        if sl_optimization_result:
            # Apply dynamic SL/TP parameters to the strategy's class attributes
            self.stoploss = sl_optimization_result.get("stoploss", self.stoploss)
            self.trailing_stop_positive_offset = sl_optimization_result.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_stop_positive = sl_optimization_result.get("trailing_stop_positive", self.trailing_stop_positive)

            # Update these learned parameters in params_manager for persistence
            asyncio.create_task(self.params_manager.update_strategy_roi_sl_params(
                strategy_id=self.name,
                new_roi=self.minimal_roi,
                new_stoploss=self.stoploss,
                new_trailing_stop_positive=self.trailing_stop_positive,
                new_trailing_stop_positive_offset=self.trailing_stop_positive_offset
            ))

            logger.info(f"TSL parameters bijgewerkt voor {pair}: Stoploss={self.stoploss:.2%}, TSL_Offset={self.trailing_stop_positive_offset:.2%}, TSL_Trigger={self.trailing_stop_positive:.2%}")

        return None # Geen onmiddellijke exit via AI

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Extra checks voor de AI-consensus en confidence vlak voor de trade.
        """
        current_confidence = self.confidence_engine.get_confidence_score(pair, self.name)
        current_bias = self.bias_reflector.get_bias_score(pair, self.name)

        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)

        if current_confidence < entry_conviction_threshold or current_bias < 0.5:
            logger.warning(f"Trade entry voor {pair} geweigerd in confirm_trade_entry. Lage confidence ({current_confidence:.2f}) of bias ({current_bias:.2f}) of onder drempel ({entry_conviction_threshold:.2f}).")
            return False

        logger.info(f"Trade entry voor {pair} bevestigd in confirm_trade_entry. Conf: {current_confidence:.2f}, Bias: {current_bias:.2f}.")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, **kwargs) -> bool:
        """
        Extra checks voor de AI-consensus en confidence vlak voor de exit.
        """
        logger.info(f"Trade exit voor {pair} bevestigd in confirm_trade_exit.")
        return True

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame: # Corrected type hint for dataframe
        """
        Freqtrade's entry trend populatie.
        De AI-gestuurde logica is in `custom_entry`. Hier zetten we een placeholder
        om `custom_entry` te triggeren.
        """
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame: # Corrected type hint for dataframe
        """
        Freqtrade's exit trend populatie.
        De AI-gestuurde logica is in `custom_exit`. Hier zetten we een placeholder
        om `custom_exit` te triggeren.
        """
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'exit_long'] = 1
        return dataframe

    def process_stopped_trade(self, pair: str, trade: Trade, order: Dict[str, Any], **kwargs) -> None:
        """
        Deze methode wordt aangeroepen nadat een trade is gesloten.
        Ideaal voor het triggeren van AI-reflectie en het bijwerken van leerbare variabelen.
        """
        profit_loss_pct = trade.calc_profit_ratio()

        trade_data = trade.to_json()
        trade_data['profit_pct'] = profit_loss_pct
        trade_data['exit_rate'] = order.get('price')
        trade_data['exit_type'] = order.get('ft_pair_exit_reason', 'unknown')

        logger.info(f"Trade gesloten voor {pair} (ID: {trade.id}). Resultaat: {profit_loss_pct:.2%}. Exit reden: {trade_data['exit_type']}. Trigger AI reflectie en leerloops.")

        # Log de trade in het eigen trade_log.json
        asyncio.create_task(self.trade_logger.log_trade(trade_data))

        # Trigger de AI-reflectie in een aparte taak zodat het de hoofdloop niet blokkeert
        candles_by_timeframe_for_reflect = self._get_all_relevant_candles_for_ai(pair)
        if not candles_by_timeframe_for_reflect:
            logger.warning(f"Geen voldoende dataframes voor post-trade reflectie voor {pair}. Reflectie overgeslagen.")
            return

        asyncio.create_task(
            self.reflectie_lus.process_reflection_cycle(
                symbol=pair,
                candles_by_timeframe=candles_by_timeframe_for_reflect,
                strategy_id=self.name,
                trade_context=trade_data,
                current_bias=self.bias_reflector.get_bias_score(pair, self.name),
                current_confidence=self.confidence_engine.get_confidence_score(pair, self.name),
                mode=self.config.get('runmode', 'live')
            )
        )

        # Update strategy performance in strategy_manager (voor mutatie en selectie)
        current_perf = self.strategy_manager.get_strategy_performance(self.name)
        new_trade_count = current_perf.get('tradeCount', 0) + 1
        new_total_profit = current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0) + profit_loss_pct # This was: current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0) * current_perf.get('tradeCount', 0) + profit_loss_pct
        new_win_rate = (current_perf.get('winRate', 0) * current_perf.get('tradeCount', 0) + (1 if profit_loss_pct > 0 else 0)) / new_trade_count if new_trade_count > 0 else 0
        new_avg_profit = new_total_profit / new_trade_count if new_trade_count > 0 else 0

        asyncio.create_task(self.strategy_manager.update_strategy_performance(
            strategy_id=self.name,
            new_performance={
                "winRate": new_win_rate,
                "avgProfit": new_avg_profit,
                "tradeCount": new_trade_count
            }
        ))

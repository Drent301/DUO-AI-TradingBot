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
from freqtrade.exchange import Exchange # For type hinting, not direct use
from freqtrade.persistence import Trade # For type hinting


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
from core.params_manager import ParamsManager # Import de nieuwe ParamsManager
from core.trade_logger import TradeLogger # Import de nieuwe TradeLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DUOAI_Strategy(IStrategy):
    """
    De hoofdstrategieklasse voor de DUO-AI Trading Bot.
    Deze strategie combineert Freqtrade met onze eigen AI-modules voor
    zelflerende besluitvorming, reflectie en optimalisatie.
    """

    # Freqtrade Hyperopt/Strategy parameters (initial defaults)
    # These are initial defaults; AI will dynamically adjust them via ParamsManager.
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
    # Deze worden eenmalig geïnitialiseerd bij het starten van de strategie.
    # Ze zijn nu toegankelijk via `self.<module_name>`.
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
    params_manager: ParamsManager = ParamsManager() # Nieuwe ParamsManager
    trade_logger: TradeLogger = TradeLogger() # Nieuwe TradeLogger


    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # Laad geleerde parameters direct bij initialisatie van de strategie
        # We roepen hier een synchrone helper aan om de async aanroep naar ParamsManager te omzeilen
        # of we zorgen dat ParamsManager's init een sync load doet.
        # ParamsManager's __init__ is al sync, dus we kunnen het direct gebruiken.
        self._load_and_apply_learned_parameters(config.get('pair_whitelist', ['N/A'])[0]) # Gebruik een initieel paar voor het laden
        logger.info(f"DUOAI_Strategy geïnitialiseerd.")

    def _load_and_apply_learned_parameters(self, pair: str) -> None:
        """
        Laadt de laatst bekende geleerde parameters van de `params_manager`
        en past ze toe op de Freqtrade strategie parameters.
        Deze methode kan ook worden aangeroepen om periodiek te updaten.
        """
        # Haal de strategie-specifieke parameters op
        strategy_params = self.params_manager.get_param(self.name, strategy_id=self.name)

        if strategy_params:
            self.minimal_roi = strategy_params.get("minimal_roi", self.minimal_roi)
            self.stoploss = strategy_params.get("stoploss", self.stoploss)
            self.trailing_stop_positive = strategy_params.get("trailing_stop_positive", self.trailing_stop_positive)
            self.trailing_stop_positive_offset = strategy_params.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)

            # Update andere geleerde parameters die direct de strategie beïnvloeden.
            # Bijv. als 'maxTradeRiskPct' dynamisch Freqtrade's stake_amount zou beïnvloeden.

            logger.debug(f"Strategie parameters voor {pair} bijgewerkt door geleerde parameters: {strategy_params}")
        else:
            logger.debug(f"Geen geleerde parameters gevonden voor {self.name}. Gebruik standaardstrategieparameters.")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Voeg indicatoren toe aan de dataframe en merge informative pairs.
        """
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

        # --- Merge informative timeframes into the main dataframe ---
        # This is the CORRECT Freqtrade way to get multi-timeframe data for your AI.
        # Make sure `informative_pairs` are correctly defined in your Freqtrade strategy configuration
        # or dynamically set if you use `self.dp`.
        for informative_tf in self.informative_timeframes:
            informative_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=informative_tf)
            if not informative_df.empty and not informative_df.equals(dataframe): # Avoid merging with itself if timeframe is the same
                dataframe = merge_informative_pair(dataframe, informative_df, self.timeframe, informative_tf, fqtrade=True)
            else:
                logger.debug(f"Informative dataframe for {metadata['pair']} on {informative_tf} is empty or same as base. Skipping merge.")


        # Haal bias en confidence op (van AI-modules)
        current_bias = self.bias_reflector.get_bias_score(metadata['pair'], self.name)
        current_confidence = self.confidence_engine.get_confidence_score(metadata['pair'], self.name)
        dataframe['ai_bias'] = current_bias
        dataframe['ai_confidence'] = current_confidence

        # Dynamische parameters laden en toepassen (bijv. entryConvictionThreshold)
        # Dit kan hier of in `custom_entry`/`custom_exit` om de meest recente waarden te garanderen.
        # Hier is het OK, als het niet te vaak wordt aangeroepen.
        self._load_and_apply_learned_parameters(metadata['pair'])

        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            leverage: float, default_stake_amount: float, **kwargs) -> float:
        """
        Pas de inzet per trade aan op basis van AI-confidence (maxTradeRiskPct).
        """
        # Krijg de max_per_trade_pct van de confidence_engine
        # Deze is dynamisch bijgewerkt door de confidence_engine.
        max_per_trade_pct_learned = self.confidence_engine.confidence_memory.get(pair, {}).get(self.name, {}).get('max_per_trade_pct', 0.1)
        # Haal ook de globale maxTradeRiskPct van de params_manager als basis
        global_max_trade_risk = self.params_manager.get_param("maxTradeRiskPct")

        # Neem het minimum van de geleerde en de globale ingestelde maxTradeRiskPct
        effective_max_trade_risk = min(max_per_trade_pct_learned, global_max_trade_risk)

        total_capital = self.wallets.get_leverage_capital(self.stake_currency)

        adjusted_stake_amount = total_capital * effective_max_trade_risk

        # Zorg ervoor dat het niet onder de minimale stake_amount komt
        # en niet hoger is dan de default_stake_amount (als die beperkt is)
        if self.stake_amount != "unlimited":
            adjusted_stake_amount = min(adjusted_stake_amount, self.stake_amount)

        # Zorg voor een minimale stake amount om fouten te voorkomen (bijv. 10 EUR)
        min_stake = 10.0 # Define a minimum stake amount, or get from config
        adjusted_stake_amount = max(adjusted_stake_amount, min_stake)

        logger.info(f"Aangepaste stake amount voor {pair}: {adjusted_stake_amount:.2f} {self.stake_currency} (gebaseerd op effectieve MaxPerTradePct: {effective_max_trade_risk:.2%}).")
        return adjusted_stake_amount

    async def custom_entry(self, pair: str, current_time: datetime,
                           dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        """
        AI-gestuurde entry-beslissing.
        Deze methode wordt door Freqtrade aangeroepen om te bepalen of een long entry moet worden geplaatst.
        """
        # Multi-timeframe data voor AI-modules:
        # De `dataframe` die hier binnenkomt, bevat al de samengevoegde informative timeframes
        # (indien correct geconfigureerd in `populate_indicators`).
        # We moeten deze opsplitsen in de `candles_by_timeframe` dict die AI-modules verwachten.

        candles_by_timeframe_for_ai: Dict[str, pd.DataFrame] = {}
        # Voeg de basis timeframe dataframe toe
        candles_by_timeframe_for_ai[self.timeframe] = dataframe.copy()

        # Extraheer informative timeframes uit de samengevoegde dataframe
        # Dit vereist dat de kolommen correct zijn benoemd door Freqtrade.
        # Bijv. '1h_open', '4h_close'. We moeten deze weer reconstrueren naar aparte DataFrames.
        # `IStrategy.extract_dataframe` helper van Freqtrade zou hier handig kunnen zijn,
        # maar die is niet standaard beschikbaar voor elke timeframe in deze context.
        # De makkelijkste manier is om `self.dp.get_pair_dataframe` te gebruiken, net als in `process_stopped_trade`.

        for tf in self.informative_timeframes:
            try:
                informative_df = self.dp.get_pair_dataframe(pair, tf)
                if not informative_df.empty:
                    candles_by_timeframe_for_ai[tf] = informative_df.copy()
            except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen via self.dp voor entry: {e}")

        # Haal de geleerde bias en confidence op
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)

        # Haal de entry conviction threshold op van params_manager
        entry_conviction_threshold = self.params_manager.get_param("entryConvictionThreshold", strategy_id=self.name)

        entry_decision = await self.entry_decider.should_enter(
            dataframe=dataframe, # Basis timeframe DF met alle indicatoren en gemergde informatives
            symbol=pair,
            current_strategy_id=self.name,
            trade_context={"current_price": dataframe['close'].iloc[-1], "timeframe": self.timeframe, "candles_by_timeframe": candles_by_timeframe_for_ai}, # Pass all relevant TFs
            # Pass learned parameters directly for the decision process
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            entry_conviction_threshold=entry_conviction_threshold
        )

        if entry_decision['enter']:
            logger.info(f"Entry toegestaan voor {pair}. Reden: {entry_decision['reason']}. Confidence: {entry_decision['confidence']:.2f}")
            return 1.0 # Return a value > 0 to indicate that an entry is desired.
                       # Freqtrade will then use the stake_amount from custom_stake_amount.

        logger.info(f"Entry geweigerd voor {pair}. Reden: {entry_decision['reason']}")
        return None # Entry niet toegestaan

    async def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                          dataframe: pd.DataFrame, **kwargs) -> Optional[float]:
        """
        AI-gestuurde exit-beslissing en dynamische SL/TP aanpassing.
        Deze methode wordt door Freqtrade aangeroepen om te bepalen of een long exit moet worden geplaatst.
        """
        # Multi-timeframe data voor AI-modules (net als bij custom_entry)
        candles_by_timeframe_for_ai: Dict[str, pd.DataFrame] = {}
        candles_by_timeframe_for_ai[self.timeframe] = dataframe.copy()
        for tf in self.informative_timeframes:
            try:
                informative_df = self.dp.get_pair_dataframe(pair, tf)
                if not informative_df.empty:
                    candles_by_timeframe_for_ai[tf] = informative_df.copy()
            except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen via self.dp voor exit: {e}")

        # Haal de geleerde bias en confidence op
        learned_bias = self.bias_reflector.get_bias_score(pair, self.name)
        learned_confidence = self.confidence_engine.get_confidence_score(pair, self.name)

        # Haal de exit conviction drop trigger op van params_manager
        exit_conviction_drop_trigger = self.params_manager.get_param("exitConvictionDropTrigger", strategy_id=self.name)


        exit_decision = await self.exit_optimizer.should_exit(
            dataframe=dataframe,
            trade=trade.to_json(), # Freqtrade trade object, converteer naar JSON voor AI context
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai, # Pass all relevant TFs
            # Pass learned parameters directly for the decision process
            learned_bias=learned_bias,
            learned_confidence=learned_confidence,
            exit_conviction_drop_trigger=exit_conviction_drop_trigger
        )

        if exit_decision['exit']:
            logger.info(f"Exit getriggerd voor {pair}. Reden: {exit_decision['reason']}. Confidence: {exit_decision['confidence']:.2f}")
            return dataframe['close'].iloc[-1] # Return the current closing price for immediate exit

        # Dynamische Trailing Stop Loss Optimalisatie (via AI)
        sl_optimization_result = await self.exit_optimizer.optimize_trailing_stop_loss(
            dataframe=dataframe,
            trade=trade.to_json(),
            symbol=pair,
            current_strategy_id=self.name,
            candles_by_timeframe=candles_by_timeframe_for_ai, # Pass all relevant TFs
            # Pass learned parameters directly for the optimization process
            learned_bias=learned_bias,
            learned_confidence=learned_confidence
        )

        if sl_optimization_result:
            # Apply dynamic SL/TP parameters to the strategy's class attributes
            self.stoploss = sl_optimization_result.get("stoploss", self.stoploss)
            self.trailing_stop_positive_offset = sl_optimization_result.get("trailing_stop_positive_offset", self.trailing_stop_positive_offset)
            self.trailing_stop_positive = sl_optimization_result.get("trailing_stop_positive", self.trailing_stop_positive)

            # Update these learned parameters in params_manager as well, so they persist
            asyncio.create_task(self.params_manager.update_strategy_roi_sl_params(
                strategy_id=self.name,
                new_roi=self.minimal_roi, # Keep current ROI, or get from AI
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

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Freqtrade's entry trend populatie.
        De AI-gestuurde logica is in `custom_entry`. Hier zetten we een placeholder
        om `custom_entry` te triggeren.
        """
        # A simple condition to ensure `custom_entry` is called.
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Freqtrade's exit trend populatie.
        De AI-gestuurde logica is in `custom_exit`. Hier zetten we een placeholder
        om `custom_exit` te triggeren.
        """
        # A simple condition to ensure `custom_exit` is called.
        dataframe.loc[
            (dataframe['volume'] > 0), # Always True if there's any volume
            'exit_long'] = 1
        return dataframe

    def process_stopped_trade(self, pair: str, trade: Trade, order: Dict[str, Any],
                              # Met order parameter, kunnen we de uiteindelijke exit_rate krijgen
                              # en of het een verlies of winst was
                              **kwargs) -> None:
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
        candles_by_timeframe_for_reflect = {}
        candles_by_timeframe_for_reflect[self.timeframe] = self.dp.get_pair_dataframe(pair, self.timeframe).copy()
        for tf in self.informative_timeframes:
             try:
                informative_df = self.dp.get_pair_dataframe(pair, tf)
                if not informative_df.empty:
                    candles_by_timeframe_for_reflect[tf] = informative_df.copy()
             except Exception as e:
                logger.warning(f"Kon informative dataframe voor {pair} op {tf} niet ophalen voor post-trade reflectie: {e}")


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
        # Corrected calculation for new_total_profit:
        old_total_profit = current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0)
        new_total_profit = old_total_profit + profit_loss_pct # Calculation was: current_perf.get('avgProfit', 0.0) * current_perf.get('tradeCount', 0) * current_perf.get('tradeCount', 0) + profit_loss_pct

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

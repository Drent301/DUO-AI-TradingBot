# --- Do not remove these libs ---
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from datetime import datetime # Import datetime
from typing import Optional # Import Optional for type hinting

from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter, \
    BooleanParameter
from freqtrade.exchange import timeframe_to_prev_date

# --- Custom Modules ---
# Hier komen de custom modules die we nodig hebben voor de AI-logica
import sys
from pathlib import Path
# Voeg de 'core' map toe aan sys.path om imports vanuit daar mogelijk te maken
core_path = Path(__file__).parent.parent / 'core'
if str(core_path) not in sys.path:
    sys.path.append(str(core_path))

try:
    from entry_decider import EntryDecider
    from exit_optimizer import ExitOptimizer
    from interval_selector import IntervalSelector
    # TODO: Importeer andere AI-gerelateerde modules zoals BiasReflector, ConfidenceEngine etc.
    # als ze direct in de strategy worden gebruikt (ipv via EntryDecider/ExitOptimizer)
except ImportError as e:
    print(f"FOUT BIJ IMPORTEREN CUSTOM MODULES IN STRATEGIE: {e}")
    # Fallback of error handling als modules niet gevonden worden
    # Dit kan betekenen dat de bot in een niet-AI modus draait of stopt.
    EntryDecider = None
    ExitOptimizer = None
    IntervalSelector = None


class DUOAI_Strategy(IStrategy):
    """
    DUOAI_Strategy
    Een Freqtrade-strategie die AI gebruikt voor entry- en exitbeslissingen.
    De strategie is ontworpen om dynamisch te leren en zich aan te passen.

    Auteur: Your Name / Your Team
    Versie: 0.1
    """

    # --- Strategie Parameters & Configuratie ---
    # Deze timeframe wordt gebruikt voor het downloaden van data, maar AI kan intern andere intervallen gebruiken.
    timeframe = '5m' # Basis timeframe

    # Stoploss: Statische stoploss als fallback. AI kan dynamische SL aanbevelen.
    # [cite: 40] TODO: AI kan SL aanpassen, maar een default is nodig. [cite: 111, 184, 253, 330, 400, 474, 542]
    stoploss = -0.10  # 10% statische stoploss

    # Trailing stop: Basisinstellingen. AI kan deze dynamisch aanpassen.
    # [cite: 38]
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False


    # ROI table: Leeg, want we vertrouwen op AI of dynamische SL voor exits.
    # Kan als fallback dienen als AI-exits falen.
    minimal_roi = {
        "0": 100 # Effectively disabled if AI is primary exit mechanism
    }

    # Hyperparameters (optioneel, kunnen door AI worden beïnvloed of gebruikt)
    # Voorbeeld: een categorische parameter voor AI-modellen of modi
    # ai_model_preference = CategoricalParameter(['performance', 'balanced', 'conservative'], default='balanced', space='buy')


    # --- Custom AI Module Initialisatie ---
    # [cite: 15] Initialiseer AI modules hier. [cite: 86, 160, 229, 306, 377, 449, 519]
    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy_identifier = self.get_strategy_name() # Unieke ID voor deze strategie instance

        # Initialiseer AI-componenten als ze succesvol zijn geïmporteerd
        if EntryDecider:
            self.entry_decider = EntryDecider()
        else:
            self.entry_decider = None
            print(f"EntryDecider niet beschikbaar voor strategie {self.strategy_identifier}.")

        if ExitOptimizer:
            self.exit_optimizer = ExitOptimizer()
        else:
            self.exit_optimizer = None
            print(f"ExitOptimizer niet beschikbaar voor strategie {self.strategy_identifier}.")

        if IntervalSelector:
            self.interval_selector = IntervalSelector()
        else:
            self.interval_selector = None
            print(f"IntervalSelector niet beschikbaar voor strategie {self.strategy_identifier}.")

        # TODO: Voeg hier logging toe om de status van AI module initialisatie te bevestigen.
        print(f"DUOAI_Strategy ({self.strategy_identifier}) geïnitialiseerd met AI modules: "
              f"EntryDecider={'Ja' if self.entry_decider else 'Nee'}, "
              f"ExitOptimizer={'Ja' if self.exit_optimizer else 'Nee'}, "
              f"IntervalSelector={'Ja' if self.interval_selector else 'Nee'}")


    # --- Data & Indicatoren ---
    # [cite: 28] Indicator-berekeningen. AI kan hierop voortbouwen. [cite: 99, 172, 241, 318, 389, 462, 531]
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Voegt technische indicatoren toe aan de dataframe.
        Deze indicatoren kunnen worden gebruikt door de AI voor besluitvorming.
        """
        # Voorbeeld: RSI, MACD, Bollinger Bands (standaard Freqtrade indicatoren)
        dataframe['rsi'] = np.nan # Placeholder, talib vult dit normaal
        dataframe['macd'] = np.nan
        dataframe['macdsignal'] = np.nan
        dataframe['macdhist'] = np.nan
        dataframe['bb_upperband'] = np.nan
        dataframe['bb_middleband'] = np.nan
        dataframe['bb_lowerband'] = np.nan

        # Probeer TA-Lib te gebruiken als het beschikbaar is (na de installatiepogingen)
        try:
            import talib.abstract as ta
            dataframe['rsi'] = ta.RSI(dataframe)
            macd = ta.MACD(dataframe)
            dataframe['macd'] = macd['macd']
            dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']
            bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
            dataframe['bb_lowerband'] = bollinger['lowerband']
            dataframe['bb_middleband'] = bollinger['middleband']
            dataframe['bb_upperband'] = bollinger['upperband']
        except ImportError:
            print("WAARSCHUWING: TA-Lib niet gevonden. Strategie draait met beperkte indicatoren (placeholders).")
        except Exception as e:
            print(f"FOUT bij berekenen TA-Lib indicatoren: {e}")


        # TODO: Voeg eventueel andere relevante indicatoren toe die de AI kan gebruiken.
        # Voorbeeld: Volume, MFI, Stochastics, ADX, etc.
        # [cite: 29] Overweeg ATR voor volatiliteit, relevant voor SL/TP. [cite: 100, 174, 242, 319, 390, 463, 532]
        # dataframe['atr'] = ta.ATR(dataframe)

        # Metadata voor AI (bijv. huidige pair, timeframe)
        # Deze dataframe.attrs worden gebruikt in de PromptBuilder en andere AI modules.
        dataframe.attrs['pair'] = metadata['pair']
        dataframe.attrs['timeframe'] = self.timeframe

        return dataframe


    # --- Entry Signalen ---
    # [cite: 31] Entry-logica, aangestuurd door EntryDecider. [cite: 102, 176, 245, 322, 393, 466, 535]
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Bepaalt entry signalen op basis van AI-advies (EntryDecider).
        """
        print(f"[{metadata['pair']}] Populaten entry trend...")
        if not self.entry_decider:
            dataframe['enter_long'] = 0
            dataframe['enter_tag'] = 'EntryDecider_uitgeschakeld'
            return dataframe

        # Asynchrone wrapper voor de AI call
        async def get_entry_signal_async(df_slice):
            # EntryDecider verwacht een enkele dataframe slice (laatste candle)
            # maar de hele dataframe wordt meegegeven voor context indien nodig (bijv. voor patronen)
            # We gebruiken hier de volledige dataframe die al indicatoren bevat.
            # De EntryDecider's should_enter methode zal intern de laatste candle of relevante data selecteren.
            return await self.entry_decider.should_enter(
                dataframe=df_slice, # Geef de volledige dataframe mee
                symbol=metadata['pair'],
                current_strategy_id=self.strategy_identifier,
                trade_context = {"stake_amount": self.config.get('stake_amount', 0)} # Voorbeeld van trade context
            )

        # Initialiseer kolommen
        dataframe['enter_long'] = 0
        dataframe['enter_tag'] = ''
        dataframe['ai_confidence'] = 0.0
        dataframe['ai_learned_bias'] = 0.0

        # We evalueren alleen de laatste paar candles om performance te sparen,
        # tenzij in backtesting/hyperopt waar we alles willen evalueren.
        # TODO: Maak dit configureerbaar of slimmer.
        eval_window = 5 # Evalueer laatste 5 candles in live/dry-run
        if self.config['runmode'] in ['backtest', 'hyperopt']:
             eval_window = len(dataframe)


        # Itereren is niet ideaal, maar voor AI calls per candle (of per paar candles) soms nodig.
        # Overweeg vectorisatie als de AI-logica dit toelaat.
        # Voor nu, evalueren we de AI voor de laatste N candles.
        # De AI call gebeurt hier per candle, wat traag kan zijn.
        # In een live scenario zou dit waarschijnlijk alleen voor de allerlaatste (complete) candle gebeuren.

        # We hebben hier de volledige dataframe nodig voor `should_enter` omdat het indicatoren van
        # voorgaande candles kan gebruiken (bijv. voor patroon detectie).
        # De AI call in `should_enter` is async, dus we moeten dit correct afhandelen.
        # We kunnen niet direct `await` gebruiken in een non-async pandas apply/loop.
        # Oplossing: Verzamel taken en run ze. (Complex voor directe dataframe populatie)
        # Simpele oplossing voor nu: evalueer alleen de laatste candle in live/dry.
        # Voor backtesting, is een langzamere iteratie soms acceptabel.

        if self.config['runmode'] not in ['backtest', 'hyperopt']:
            # Live/Dry-run: alleen laatste (complete) candle
            if len(dataframe) > 0:
                try:
                    # Maak een copy van de relevante slice om SettingWithCopyWarning te voorkomen
                    # Hoewel we hier de hele dataframe meegeven, is het de bedoeling dat de AI zich focust op de laatste candle.
                    df_slice = dataframe.copy() # Geef de hele dataframe mee voor context

                    # Roep de async methode aan en wacht op het resultaat
                    # Let op: dit blokkeert tot de AI call klaar is.
                    # In een live bot is dit riskant als de AI lang duurt.
                    # TODO: Overweeg een non-blocking manier of een timeout.
                    signal_result = asyncio.run(get_entry_signal_async(df_slice))

                    if signal_result and signal_result.get('enter'):
                        dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = signal_result.get('reason', 'AI_entry')
                        dataframe.loc[dataframe.index[-1], 'ai_confidence'] = signal_result.get('confidence', 0.0)
                        dataframe.loc[dataframe.index[-1], 'ai_learned_bias'] = signal_result.get('learned_bias', 0.0)
                    else:
                        dataframe.loc[dataframe.index[-1], 'enter_tag'] = signal_result.get('reason', 'AI_no_entry')
                        dataframe.loc[dataframe.index[-1], 'ai_confidence'] = signal_result.get('confidence', 0.0)
                        dataframe.loc[dataframe.index[-1], 'ai_learned_bias'] = signal_result.get('learned_bias', 0.0)

                except Exception as e:
                    print(f"FOUT bij AI entry evaluatie voor {metadata['pair']}: {e}")
                    dataframe.loc[dataframe.index[-1], 'enter_tag'] = 'AI_eval_error'
        else:
            # Backtesting/Hyperopt: evalueer voor een window (of alles)
            # Dit is nog steeds suboptimaal en traag.
            # Een betere aanpak voor backtesting zou zijn om de AI calls te batchen of
            # de AI logica te vectoriseren indien mogelijk.
            for i in range(max(0, len(dataframe) - eval_window), len(dataframe)):
                try:
                    df_slice = dataframe.iloc[:i+1].copy() # Data tot en met huidige candle
                    signal_result = asyncio.run(get_entry_signal_async(df_slice))
                    if signal_result and signal_result.get('enter'):
                        dataframe.loc[dataframe.index[i], 'enter_long'] = 1
                        dataframe.loc[dataframe.index[i], 'enter_tag'] = signal_result.get('reason', 'AI_entry')
                        dataframe.loc[dataframe.index[i], 'ai_confidence'] = signal_result.get('confidence', 0.0)
                        dataframe.loc[dataframe.index[i], 'ai_learned_bias'] = signal_result.get('learned_bias', 0.0)
                    else:
                        dataframe.loc[dataframe.index[i], 'enter_tag'] = signal_result.get('reason', 'AI_no_entry')
                        dataframe.loc[dataframe.index[i], 'ai_confidence'] = signal_result.get('confidence', 0.0)
                        dataframe.loc[dataframe.index[i], 'ai_learned_bias'] = signal_result.get('learned_bias', 0.0)
                except Exception as e:
                    print(f"FOUT bij AI entry evaluatie (backtest) voor {metadata['pair']} op index {i}: {e}")
                    dataframe.loc[dataframe.index[i], 'enter_tag'] = 'AI_eval_error_backtest'
        return dataframe


    # --- Exit Signalen ---
    # [cite: 34] Exit-logica, aangestuurd door ExitOptimizer. [cite: 105, 179, 248, 325, 396, 469, 538]
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Bepaalt exit signalen op basis van AI-advies (ExitOptimizer).
        """
        print(f"[{metadata['pair']}] Populaten exit trend...")
        if not self.exit_optimizer:
            dataframe['exit_long'] = 0
            dataframe['exit_tag'] = 'ExitOptimizer_uitgeschakeld'
            return dataframe

        # Vergelijkbaar met entry, AI exit evaluatie.
        # De `custom_exit` methode wordt gebruikt voor actieve trades.
        # Hier kunnen we algemene 'exit signalen' genereren als de strategie dit vereist
        # (bijv. voor `exit_long=1` op basis van marktcondities, niet per se een open trade).
        # Voor nu laten we dit leeg, omdat `custom_exit` de exits per trade afhandelt.
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ''
        return dataframe

    # --- Custom Exit Logica (per trade) ---
    # [cite: 35] Dynamische SL/TP via ExitOptimizer. [cite: 106, 180, 249, 326, 397, 470, 539]
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Custom exit methode om AI-gestuurde exits per trade te bepalen.
        """
        if not self.exit_optimizer:
            return None # Geen custom exit als optimizer niet beschikbaar is

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        # Converteer Freqtrade trade object naar een dictionary die ExitOptimizer verwacht
        # Dit is een vereenvoudiging; mogelijk moeten meer velden worden gemapt.
        trade_dict = {
            "id": trade.id,
            "pair": trade.pair,
            "is_open": trade.is_open,
            "profit_pct": current_profit, # Freqtrade geeft dit al als percentage (0.1 = 10%)
            "stake_amount": trade.stake_amount,
            "open_date": trade.open_date_utc, # Zorg dat dit UTC is
            "open_rate": trade.open_rate,
            "current_rate": current_rate,
            # TODO: Voeg eventuele andere relevante trade info toe
        }

        # Roep ExitOptimizer.should_exit aan
        # Dit is een async call, dus we moeten het synchroon uitvoeren in deze Freqtrade callback.
        # Let op: Dit blokkeert!
        try:
            exit_signal_result = asyncio.run(self.exit_optimizer.should_exit(
                dataframe=dataframe.copy(), # Geef een copy mee
                trade=trade_dict,
                symbol=pair,
                current_strategy_id=self.strategy_identifier
            ))

            if exit_signal_result and exit_signal_result.get('exit'):
                print(f"Custom AI exit voor {pair}: {exit_signal_result.get('reason', 'AI_exit')}")
                return exit_signal_result.get('reason', 'AI_custom_exit') # Geeft de reden voor de exit

        except Exception as e:
            print(f"FOUT bij AI custom_exit evaluatie voor {pair}: {e}")
            return "AI_exit_eval_error" # Geeft aan dat er een fout was, maar exit niet per se

        return None # Geen custom AI exit signaal


    # --- Custom Stoploss Logica ---
    # [cite: 40] AI kan SL aanpassen. [cite: 111, 184, 253, 330, 400, 474, 542]
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        """
        Berekent dynamische stoploss op basis van AI-advies (ExitOptimizer).
        Retourneert de absolute stoploss prijs.
        """
        if not self.exit_optimizer:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        trade_dict = {
            "id": trade.id, "pair": trade.pair, "is_open": trade.is_open,
            "profit_pct": current_profit, "stake_amount": trade.stake_amount,
            "open_date": trade.open_date_utc, "open_rate": trade.open_rate,
            "current_rate": current_rate,
        }

        try:
            # Optimaliseer SL parameters met AI
            # Dit retourneert een dict met bijv. "stoploss", "trailing_stop_positive_offset"
            sl_params = asyncio.run(self.exit_optimizer.optimize_trailing_stop_loss(
                dataframe=dataframe.copy(),
                trade=trade_dict,
                symbol=pair,
                current_strategy_id=self.strategy_identifier
            ))

            if sl_params:
                # Update Freqtrade's dynamische stoploss instellingen voor deze trade
                # Dit is conceptueel; Freqtrade's `custom_stoploss` retourneert de SL prijs.
                # Het direct aanpassen van trailing stop parameters is complexer en vereist
                # mogelijk interactie met `ITrade` in `execute_sell` of via `adjust_trade_position`.
                # Voor nu, focussen we op het retourneren van een AI-gebaseerde harde stoploss.

                if 'stoploss' in sl_params and sl_params['stoploss'] is not None:
                    # De geretourneerde 'stoploss' is een percentage (bijv. -0.05 voor -5%)
                    # We moeten dit omzetten naar een absolute prijs.
                    # Freqtrade verwacht dat custom_stoploss de absolute prijs retourneert.
                    # Een negatieve waarde van custom_stoploss wordt geïnterpreteerd als "geen custom stoploss".
                    # Dus, als we een stop willen, moet de geretourneerde waarde > 0 zijn.

                    # Bereken de absolute stoploss prijs op basis van de entry prijs
                    # en het door AI aanbevolen percentage.
                    # Let op: Freqtrade's `stoploss` parameter is negatief.
                    # Als sl_params['stoploss'] = -0.05 (AI wil 5% SL onder entry),
                    # dan is de prijs: trade.open_rate * (1 + sl_params['stoploss'])
                    #                     = trade.open_rate * (1 - 0.05)
                    #                     = trade.open_rate * 0.95

                    # Belangrijk: `custom_stoploss` moet de SL prijs retourneren.
                    # Een waarde < `current_rate` voor LONG trades.
                    # Een waarde > `current_rate` voor SHORT trades (indien ondersteund).

                    # Voorbeeld: AI stelt een stoploss percentage voor (bijv. -0.03 = 3% onder open_rate)
                    ai_sl_percentage = sl_params['stoploss']
                    if ai_sl_percentage < 0: # Moet negatief zijn voor een stop loss
                        # Bereken de absolute stoploss prijs
                        # Voor een LONG trade:
                        stop_price = trade.open_rate * (1 + ai_sl_percentage) # bijv. 100 * (1 - 0.03) = 97

                        # Zorg ervoor dat de stoploss niet boven de huidige rate wordt gezet (tenzij het een take-profit is)
                        # en niet te ver weg als de trade al winstgevend is (laat trailing stop dat afhandelen)
                        # Dit is een harde stoploss, dus het moet onder de entryprijs liggen.
                        if stop_price < trade.open_rate and stop_price < current_rate : # Alleen als het een daadwerkelijke stop is
                           print(f"AI custom stoploss voor {pair} @ {stop_price:.5f} (AI SL: {ai_sl_percentage:.2%})")
                           return stop_price

                # TODO: Werk de logica voor trailing stop aanpassingen hier verder uit.
                # Dit kan complex zijn omdat Freqtrade's `custom_stoploss` primair een harde SL prijs verwacht.
                # Het aanpassen van trailing parameters on-the-fly voor een specifieke trade
                # is niet direct ondersteund via deze callback op dezelfde manier.
                # Men zou `trade.adjust_stoploss()` kunnen overwegen, maar dat is vanuit een andere context.
                # Een workaround kan zijn om de `trailing_stop` attributen van de strategie hier te beïnvloeden,
                # maar dat geldt dan globaal voor nieuwe trades of moet slim worden gefilterd.

        except Exception as e:
            print(f"FOUT bij AI custom_stoploss evaluatie voor {pair}: {e}")

        return None # Geen AI-gebaseerde custom stoploss, Freqtrade gebruikt de default.


    # --- Optioneel: Timeframe optimalisatie ---
    # [cite: 10] AI kan het beste timeframe selecteren. [cite: 81, 155, 224, 301, 372, 444, 515]
    def bot_loop_start(self, **kwargs) -> None:
        """
        Wordt aangeroepen aan het begin van elke bot iteratie.
        Kan worden gebruikt om dynamisch het timeframe aan te passen.
        """
        if not self.interval_selector:
            return

        # Voorbeeld: Eens per uur het beste interval opnieuw evalueren voor een specifieke base_token
        # Dit is een placeholder; de logica voor wanneer en hoe vaak dit gebeurt, moet worden verfijnd.
        # We hebben een manier nodig om de 'current_pair' of een representatieve pair te krijgen.
        # Laten we aannemen dat we dit doen voor de eerste pair in de whitelist voor nu.

        # TODO: Dit is een zeer gesimplificeerde trigger. Moet veel robuuster.
        # if datetime.now().minute == 0: # Eens per uur
        #     try:
        #         # Haal een representatieve pair op (bijv. eerste van de whitelist)
        #         # Dit is een zwak punt; hoe kies je de 'base_token' en 'token' voor algemene interval selectie?
        #         # Mogelijk moet dit per pair of per base currency.
        #         # Voor nu, hardcoded voorbeeld:
        #         example_pair = self.config['exchange']['pair_whitelist'][0] # Bv. 'ETH/USDT'
        #         base_currency = example_pair.split('/')[1] # USDT
        #         main_currency = example_pair.split('/')[0] # ETH

        #         if base_currency in self.interval_selector.SUPPORTED_BASES: # Check of base ondersteund wordt
        #             print(f"[{self.strategy_identifier}] Evalueren beste interval voor {main_currency} based trades...")
        #             best_interval = asyncio.run(self.interval_selector.detect_best_interval(
        #                 base_token=base_currency,
        #                 strategy=self.strategy_identifier,
        #                 token=main_currency # Of de volledige pair? Hangt af van implementatie.
        #             ))
        #             if best_interval and best_interval != self.timeframe:
        #                 print(f"AI stelt nieuw timeframe voor: {best_interval}. HUIDIGE STRATEGIE KAN DIT NIET DYNAMISCH WIJZIGEN.")
        #                 # OPMERKING: Freqtrade staat niet toe om `self.timeframe` dynamisch per strategie-instance te wijzigen
        #                 # nadat de bot is gestart. Dit zou een globale aanpassing vereisen of een complexere setup
        #                 # met meerdere bot instances of strategieën per timeframe.
        #                 # Deze logica is dus meer voor 'inzicht' of voor een systeem dat de bot herconfigureert.
        #     except Exception as e:
        #         print(f"FOUT bij AI interval evaluatie: {e}")
        pass # Voor nu, geen actie in bot_loop_start.

# core/reflectie_lus.py
import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd # Voor het verwerken van Freqtrade DataFrames
import numpy as np # Toegevoegd voor mock data generatie

# Importeer je eigen modules
from core.gpt_reflector import GPTReflector
from core.grok_reflector import GrokReflector
from core.prompt_builder import PromptBuilder
# Import reflectie_analyser, assuming it's in the same directory
# If reflectie_analyser.py contains those functions directly:
from core.reflectie_analyser import analyseReflecties, generateMutationProposal, analyzeTimeframeBias
from core.bias_reflector import BiasReflector
from core.confidence_engine import ConfidenceEngine

# from core.entry_decider import EntryDecider # Nog te implementeren
# from core.exit_optimizer import ExitOptimizer # Nog te implementeren
import dotenv # Toegevoegd voor de __main__ sectie

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Padconfiguratie
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
LOG_FILE = os.path.join(MEMORY_DIR, 'reflectie-logboek.json')
# cache_Dir and cache_file for live_search_cache are now handled within grok_sentiment_fetcher.py

# Zorg dat de memory map bestaat
os.makedirs(MEMORY_DIR, exist_ok=True)

class ReflectieLus:
    """
    Centrale reflectielus voor de DUO-AI Trading Bot.
    Orkestreert de dataverzameling, promptgeneratie, AI-reflectie en opslag van resultaten.
    Vertaald en geoptimaliseerd van reflectieLus.js en aiActivationEngine.js.
    """

    def __init__(self):
        self.gpt_reflector = GPTReflector()
        self.grok_reflector = GrokReflector()
        self.prompt_builder = PromptBuilder() # This will fail if PromptBuilder is not defined
        self.last_reflection_timestamps: Dict[str, float] = {} # Per token, voor AI activatie trigger
        self.analysis_result_placeholder = {"summary": "Analysis not fully implemented"} # Placeholder
        self.mutation_proposal_placeholder = {"action": "No mutation proposed yet"} # Placeholder


        # Initialiseer logboekbestand als het niet bestaat
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        logger.info(f"ReflectieLus geïnitialiseerd. Logboek: {LOG_FILE}")

    async def _store_reflection(self, reflection: Dict[str, Any]) -> bool:
        """
        Slaat een reflectie-object op in het logboekbestand.
        Vertaald van storeReflectie in reflectieLus.js.
        """
        if not reflection or not isinstance(reflection, dict):
            logger.debug('[StoreReflection] Ongeldige reflectie om op te slaan.')
            return False
        try:
            logs = []
            if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                        if not isinstance(logs, list): # Ensure it's a list
                            logs = []
                    except json.JSONDecodeError:
                        logs = [] # Bestand leeg of ongeldig, begin met lege lijst

            logs.append({**reflection, "timestamp": datetime.now().isoformat()})

            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)

            logger.debug(f'[StoreReflection] Reflectie opgeslagen voor {reflection.get("token", "N/A")}.')
            return True
        except Exception as e:
            logger.warning(f'[StoreReflection] Fout bij opslaan: {e}')
            return False

    def _update_last_reflection_timestamp(self, token: str):
        """
        Update het timestamp van de laatste reflectie voor een token.
        Vertaald van updateLastReflection in aiActivationEngine.js.
        """
        self.last_reflection_timestamps[token] = datetime.now().timestamp() * 1000

    async def process_reflection_cycle(
        self,
        symbol: str,
        candles_by_timeframe: Dict[str, pd.DataFrame],
        strategy_id: str,
        trade_context: Optional[Dict[str, Any]] = None, # Context van de trade (entry/exit)
        current_bias: float = 0.5, # Huidige bias van de strategie
        current_confidence: float = 0.5, # Huidige confidence van de strategie
        mode: str = 'live', # 'live', 'dry_run', 'backtest', 'pretrain'
        prompt_type: str = 'marketAnalysis',
        pattern_data: Optional[Dict[str, Any]] = None,
        bias_reflector_instance: Optional[BiasReflector] = None, # New
        confidence_engine_instance: Optional[ConfidenceEngine] = None # New
    ) -> Optional[Dict[str, Any]]:
        """
        Verwerkt een enkele reflectiecyclus voor een gegeven symbool.
        Dit is de kernlogica van reflectieLus.js (processReflectieCycle) en aiActivationEngine.js (activateAI).
        """
        if trade_context is None:
            trade_context = {}

        logger.info(f"[ReflectieCyclus] Starten reflectiecyclus voor {symbol} ({strategy_id}) in mode: {mode} met prompt type: {prompt_type}")

        # 1. Gegevensverzameling
        # Freqtrade zal de DataFrames van candles_by_timeframe leveren
        # Sentiment data wordt opgehaald door prompt_builder via grok_sentiment_fetcher
        # Patronen worden gedetecteerd door prompt_builder via cnn_patterns

        # 2. Prompt Generatie
        # De prompt_builder haalt al de sentiment en patroondata op intern.
        # This will fail if PromptBuilder or its methods are not defined/implemented

        # Prepare additional_context for prompt builder
        additional_context_for_prompt = {
            **(trade_context or {}), # Original trade_context items
            'pattern_data': pattern_data, # Add pattern_data
            # social_sentiment_grok would be in trade_context if added by AIActivationEngine / strategy
        }

        try:
            prompt = await self.prompt_builder.generate_prompt_with_data(
                candles_by_timeframe=candles_by_timeframe,
                symbol=symbol,
                prompt_type=prompt_type, # Use the passed prompt_type
                current_bias=current_bias,
                current_confidence=current_confidence,
                additional_context=additional_context_for_prompt
            )
        except AttributeError as e:
            logger.error(f"PromptBuilder of methode niet gevonden: {e}. Implementeer PromptBuilder eerst.")
            prompt = f"Dummy prompt for {symbol} due to PromptBuilder error." # Fallback dummy prompt
        except Exception as e:
            logger.error(f"Fout tijdens prompt generatie: {e}")
            prompt = f"Dummy prompt for {symbol} due to error: {e}"


        if not prompt:
            logger.warning(f"[ReflectieCyclus] Geen prompt gegenereerd voor {symbol}.")
            return None

        # 3. AI Reflectie aanroepen
        gpt_result = await self.gpt_reflector.ask_ai(prompt, context={"symbol": symbol, "strategy_id": strategy_id, **trade_context})
        grok_result = await self.grok_reflector.ask_grok(prompt, context={"symbol": symbol, "strategy_id": strategy_id, **trade_context})

        if not gpt_result and not grok_result:
            logger.error(f"[ReflectieCyclus] Geen AI-reflectie ontvangen van GPT of Grok voor {symbol}.")
            return None

        # Handle cases where one or both results might be None or empty
        gpt_conf = gpt_result.get('confidence', 0) if gpt_result else 0
        grok_conf = grok_result.get('confidence', 0) if grok_result else 0
        gpt_bias = gpt_result.get('bias', 0.5) if gpt_result else 0.5 # Assuming bias is a float, default to neutral
        grok_bias = grok_result.get('bias', 0.5) if grok_result else 0.5

        num_valid_results = sum(1 for res in [gpt_result, grok_result] if res and res.get('confidence') is not None)
        if num_valid_results == 0: num_valid_results = 1 # Avoid division by zero if both fail

        combined_confidence = ( (gpt_conf or 0) + (grok_conf or 0) ) / num_valid_results
        combined_bias = ( (gpt_bias or 0.5) + (grok_bias or 0.5) ) / num_valid_results


        combined_reflection = f"GPT: {gpt_result.get('reflectie', 'N/A') if gpt_result else 'N/A'}
Grok: {grok_result.get('reflectie', 'N/A') if grok_result else 'N/A'}"
        combined_intentie = (gpt_result.get('intentie') if gpt_result else None) or                             (grok_result.get('intentie') if grok_result else None) or 'N/A'
        combined_emotie = (gpt_result.get('emotie') if gpt_result else None) or                           (grok_result.get('emotie') if grok_result else None) or 'N/A'

        # Determine Sentiment Data Status and Content for logging
        sentiment_data_status = "absent"
        sentiment_items_for_log = []
        social_sentiment_grok_data = additional_context_for_prompt.get('social_sentiment_grok')

        if social_sentiment_grok_data is not None: # Key is present
            if isinstance(social_sentiment_grok_data, list):
                if social_sentiment_grok_data: # List is not empty
                    sentiment_data_status = "present_and_not_empty"
                    sentiment_items_for_log = [item.get('sentiment', 'unknown') for item in social_sentiment_grok_data]
                else: # List is empty
                    sentiment_data_status = "present_but_empty"
            else: # Not a list (e.g., could be None if key exists with None value, or other type)
                sentiment_data_status = "present_malformed_or_unexpected"

        # Evaluate Basic Sentiment-Trade Correlation
        sentiment_trade_correlation_notes = "N/A"
        if sentiment_data_status == "present_and_not_empty" and trade_context:
            trade_is_profitable = trade_context.get('profit_pct', 0) > 0
            first_sentiment = social_sentiment_grok_data[0].get('sentiment', 'unknown') # Already checked not empty

            outcome_str = "Profitable" if trade_is_profitable else "Not Profitable"
            simplified_correlation = "N/A"

            if first_sentiment == "positive":
                simplified_correlation = "positive_sentiment_aligned_with_profit" if trade_is_profitable else "positive_sentiment_misaligned_with_profit"
            elif first_sentiment == "negative":
                simplified_correlation = "negative_sentiment_aligned_with_loss" if not trade_is_profitable else "negative_sentiment_misaligned_with_loss"
            elif first_sentiment == "neutral":
                simplified_correlation = f"neutral_sentiment_outcome_{outcome_str.lower().replace(' ', '_')}"
            else: # unknown sentiment
                simplified_correlation = f"unknown_sentiment_({first_sentiment})_outcome_{outcome_str.lower().replace(' ', '_')}"

            sentiment_trade_correlation_notes = f"Trade: {outcome_str}. First sentiment: {first_sentiment}. Correlation: {simplified_correlation}. All sentiments: {sentiment_items_for_log}"

        # 4. Reflectie Loggen
        reflection_log_entry = {
            "token": symbol,
            "strategyId": strategy_id,
            "prompt": prompt,
            "gpt_response": gpt_result if gpt_result else {},
            "grok_response": grok_result if grok_result else {},
            "combined_confidence": combined_confidence,
            "combined_bias": combined_bias,
            "combined_intentie": combined_intentie,
            "combined_emotie": combined_emotie,
            "trade_context": trade_context,
            "mode": mode,
            "sentiment_data_status": sentiment_data_status,
            "sentiment_items_logged": sentiment_items_for_log,
            "sentiment_trade_correlation_notes": sentiment_trade_correlation_notes
        }
        await self._store_reflection(reflection_log_entry)
        self._update_last_reflection_timestamp(symbol)
        logger.info(f"[ReflectieCyclus] Reflectiecyclus voltooid voor {symbol} ({strategy_id}). Log opgeslagen.")

        # 5. Update Bias and Confidence based on reflection results
        if bias_reflector_instance:
            trade_profit_pct = trade_context.get('profit_pct') # Can be None
            try:
                # Extract data needed for the enhanced update_bias call
                sentiment_items_logged = reflection_log_entry.get('sentiment_items_logged', [])
                first_sentiment_observed = sentiment_items_logged[0] if sentiment_items_logged else None
                trade_direction = trade_context.get('trade_direction') # Assumes 'trade_direction' is in trade_context

                await bias_reflector_instance.update_bias(
                    token=symbol,
                    strategy_id=strategy_id,
                    new_ai_bias=reflection_log_entry['combined_bias'],
                    confidence=reflection_log_entry['combined_confidence'],
                    trade_result_pct=trade_profit_pct,
                    sentiment_data_status=reflection_log_entry.get('sentiment_data_status'),
                    first_sentiment_observed=first_sentiment_observed,
                    trade_direction=trade_direction
                )
                logger.info(f"[ReflectieCyclus] Called BiasReflector.update_bias for {symbol}/{strategy_id} with sentiment details.")
            except Exception as e:
                logger.error(f"[ReflectieCyclus] Error calling BiasReflector.update_bias for {symbol}/{strategy_id}: {e}")

        if confidence_engine_instance:
            trade_profit_pct = trade_context.get('profit_pct') # Can be None
            try:
                # Ensure first_sentiment_observed and trade_direction are available here
                # These are the same variables prepared for the BiasReflector call earlier in the code.
                # sentiment_data_status is directly from reflection_log_entry
                # first_sentiment_observed was from sentiment_items_logged (from reflection_log_entry)
                # trade_direction was from trade_context

                # Re-fetch them here to be explicit about their source for this call,
                # or ensure they are passed down if scope changes.
                # For now, assuming they are still in scope from the BiasReflector preparation block.
                # The variables `first_sentiment_observed` and `trade_direction` are defined
                # just before the `bias_reflector_instance.update_bias` call.

                current_sentiment_data_status = reflection_log_entry.get('sentiment_data_status')
                # `first_sentiment_observed` and `trade_direction` should be in scope from BiasReflector section
                # If not, they need to be re-derived:
                # sentiment_items_logged_for_conf = reflection_log_entry.get('sentiment_items_logged', [])
                # first_sentiment_observed_for_conf = sentiment_items_logged_for_conf[0] if sentiment_items_logged_for_conf else None
                # trade_direction_for_conf = trade_context.get('trade_direction')

                await confidence_engine_instance.update_confidence(
                    token=symbol,
                    strategy_id=strategy_id,
                    ai_reported_confidence=reflection_log_entry['combined_confidence'],
                    trade_result_pct=trade_profit_pct,
                    sentiment_data_status=current_sentiment_data_status, # from reflection_log_entry
                    first_sentiment_observed=first_sentiment_observed, # from earlier block
                    trade_direction=trade_direction # from earlier block
                )
                logger.info(f"[ReflectieCyclus] Called ConfidenceEngine.update_confidence for {symbol}/{strategy_id} with sentiment details (Status: {current_sentiment_data_status}, FirstObs: {first_sentiment_observed}, Direction: {trade_direction}).")
            except Exception as e:
                logger.error(f"[ReflectieCyclus] Error calling ConfidenceEngine.update_confidence for {symbol}/{strategy_id}: {e}")

        # 6. Verdere Analyse & Aanpassing (via reflectie_analyser en andere modules)
        logger.info(f"[ReflectieCyclus] Starten van verdere analyse voor {symbol} ({strategy_id}).")
        all_logs = []
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    all_logs = json.load(f)
                    if not isinstance(all_logs, list): all_logs = []
            except json.JSONDecodeError:
                logger.warning(f"Reflectie logboek {LOG_FILE} is leeg of corrupt. Kan geen analyse uitvoeren.")
                all_logs = []

        analysis_result = self.analysis_result_placeholder
        mutation_proposal = self.mutation_proposal_placeholder


        if all_logs:
            relevant_reflections = [
                log for log in all_logs
                if log.get('token') == symbol and log.get('strategyId') == strategy_id
            ]

            if relevant_reflections:
                try:
                    analysis_result = analyseReflecties(relevant_reflections)
                    timeframe_bias_analysis = analyzeTimeframeBias(relevant_reflections)

                    dummy_performance = {"winRate": 0.0, "avgProfit": 0.0, "tradeCount": 0}
                    if 'profit_pct' in trade_context:
                        dummy_performance['winRate'] = 1.0 if trade_context['profit_pct'] > 0 else 0.0
                        dummy_performance['avgProfit'] = trade_context['profit_pct']
                        dummy_performance['tradeCount'] = 1

                    current_strategy_params = {
                        "id": strategy_id,
                        "parameters": { "emaPeriod": 20, "rsiThreshold": 70 }
                    }
                    mutation_proposal = generateMutationProposal(
                        current_strategy_params,
                        combined_bias,
                        dummy_performance,
                        timeframe_bias_analysis
                    )
                    logger.info(f"Mutatievoorstel voor {symbol} ({strategy_id}): {mutation_proposal}")
                except Exception as e:
                    logger.error(f"Fout tijdens analyse of mutatievoorstel generatie: {e}")
                    analysis_result = {"summary": f"Analysefout: {e}"}
                    mutation_proposal = {"action": f"Mutatiefout: {e}"}


        return {
            "symbol": symbol,
            "strategyId": strategy_id,
            "gpt_result": gpt_result if gpt_result else {},
            "grok_result": grok_result if grok_result else {},
            "analysis_summary": analysis_result.get('summary', "Geen analyse beschikbaar"),
            "mutation_proposal": mutation_proposal
        }

    async def start_reflection_loop(self, symbols: List[str], interval_minutes: int = 5):
        """
        Start een periodieke reflectiecyclus voor de opgegeven symbolen.
        Dit is meer een daemon-achtige loop dan direct gekoppeld aan Freqtrade's events.
        """
        logger.info(f"Start reflectie loop voor {symbols} elke {interval_minutes} minuten.")

        async def run_cycle_for_all_symbols():
            for symbol in symbols:
                try:
                    mock_candles_by_timeframe = self._create_mock_dataframe_for_reflection(symbol)
                    if not mock_candles_by_timeframe:
                        logger.warning(f"Geen mock dataframes beschikbaar voor {symbol}. Sla reflectie over.")
                        continue

                    await self.process_reflection_cycle(
                        symbol=symbol,
                        candles_by_timeframe=mock_candles_by_timeframe,
                        strategy_id="DUOAI_Strategy",
                        mode='live'
                    )
                except Exception as e:
                    logger.error(f"Fout in reflectiecyclus voor {symbol}: {e}", exc_info=True)

        # Voer de eerste cyclus direct uit
        await run_cycle_for_all_symbols()

        # Plan periodieke uitvoeringen
        while True:
            await asyncio.sleep(interval_minutes * 60) # Wacht X minuten
            await run_cycle_for_all_symbols()

    def _create_mock_dataframe_for_reflection(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Hulpfunctie om mock DataFrames te creëren voor reflectiecyclus testen.
        Dit is een placeholder voor de echte DataFrames die Freqtrade zou leveren.
        """
        mock_dataframes = {}
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        # Ensure numpy is imported if not already at the top level of the file
        # import numpy as np # Already imported at the top


        for tf in timeframes:
            num_candles = 100
            data = []
            now = datetime.utcnow() # Use UTC
            interval_seconds_map = {
                '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
                '4h': 14400, '12h': 43200, '1d': 86400
            }
            interval_seconds = interval_seconds_map.get(tf, 300)

            for i in range(num_candles):
                date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
                open_ = 100 + i * 0.1 + np.random.rand() * 2
                close_ = open_ + (np.random.rand() - 0.5) * 5
                high_ = max(open_, close_) + np.random.rand() * 2
                low_ = min(open_, close_) - np.random.rand() * 2
                volume = 1000 + np.random.rand() * 500

                rsi = 50 + (np.random.rand() - 0.5) * 30
                macd_val = (np.random.rand() - 0.5) * 0.1
                macdsignal_val = macd_val * (0.8 + np.random.rand() * 0.2)
                macdhist_val = macd_val - macdsignal_val

                sma_period = 20
                std_dev_multiplier = 2
                if i >= sma_period -1 and len(data) >= sma_period -1:
                    # Ensure we use 'close' which is at index 4
                    recent_closes_for_bb = [r[4] for r in data[-(sma_period-1):]] + [close_]
                    if len(recent_closes_for_bb) == sma_period:
                        sma = np.mean(recent_closes_for_bb)
                        std_dev = np.std(recent_closes_for_bb)
                        bb_middle = sma
                        bb_upper = sma + std_dev_multiplier * std_dev
                        bb_lower = sma - std_dev_multiplier * std_dev
                    else: # Not enough data yet for this specific candle's BB
                        bb_middle = open_
                        bb_upper = high_
                        bb_lower = low_
                else:
                    bb_middle = open_
                    bb_upper = high_
                    bb_lower = low_


                data.append([date, open_, high_, low_, close_, volume, rsi, macd_val, macdsignal_val, macdhist_val, bb_upper, bb_middle, bb_lower])

            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_upperband', 'bb_middleband', 'bb_lowerband'])
            df['date'] = pd.to_datetime(df['date']) # Ensure date column is datetime
            df.set_index('date', inplace=True)
            df.attrs['timeframe'] = tf
            mock_dataframes[tf] = df

        return mock_dataframes


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)


    async def run_test_reflection_loop():
        # Setup basic logging for the test
        # Add sys import for logging handler
        import sys
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])



        ref_lus = ReflectieLus()
        test_symbols = ["ETH/USDT"]

        mock_trade_context = {
            "entry_price": 2500,
            "exit_price": 2550,
            "profit_pct": 0.02, # Profitable trade
            "trade_id": "mock_trade_123",
            "social_sentiment_grok": [ # Add this
                {"source": "TestFeed", "text": "Sample positive news for ETH.", "sentiment": "positive"},
                {"source": "AnotherFeed", "text": "Another positive one.", "sentiment": "positive"}
            ]
        }

        print("\n--- Testen van process_reflection_cycle direct (Positive Sentiment / Profit) ---")
        mock_candles = ref_lus._create_mock_dataframe_for_reflection("ETH/USDT")

        result = await ref_lus.process_reflection_cycle(
            symbol="ETH/USDT",
            candles_by_timeframe=mock_candles,
            strategy_id="DUOAI_Strategy",
            trade_context=mock_trade_context,
            current_bias=0.6,
            current_confidence=0.7,
            mode='backtest',
            prompt_type='comprehensive_analysis', # Added
            pattern_data=None # Added, or use mock_pattern_data
        )
        print("\nResultaat van directe reflectiecyclus:")
        # Use default=str to handle any non-serializable objects like datetime
        print(json.dumps(result, indent=2, default=str))

        print("\n--- Testen van process_reflection_cycle direct (Negative Sentiment / Loss) ---")
        mock_trade_context_loss = {
            "entry_price": 2500,
            "exit_price": 2450,
            "profit_pct": -0.02, # Losing trade
            "trade_id": "mock_trade_456",
            "social_sentiment_grok": [
                {"source": "TestFeed", "text": "Sample negative news for ETH.", "sentiment": "negative"}
            ]
        }
        result_loss = await ref_lus.process_reflection_cycle(
            symbol="ETH/USDT",
            candles_by_timeframe=mock_candles, # re-use
            strategy_id="DUOAI_Strategy",
            trade_context=mock_trade_context_loss, # use new context
            current_bias=0.4,
            current_confidence=0.6,
            mode='backtest',
            prompt_type='comprehensive_analysis',
            pattern_data=None
        )
        print("\nResultaat van reflectiecyclus (Negative Sentiment / Loss):")
        print(json.dumps(result_loss, indent=2, default=str))

        print("\n--- Testen van process_reflection_cycle direct (Neutral Sentiment / Profit) ---")
        mock_trade_context_neutral_profit = {
            "entry_price": 2500,
            "exit_price": 2550,
            "profit_pct": 0.02, # Profitable trade
            "trade_id": "mock_trade_789",
            "social_sentiment_grok": [
                {"source": "NeutralNews", "text": "Market is stable.", "sentiment": "neutral"}
            ]
        }
        result_neutral_profit = await ref_lus.process_reflection_cycle(
            symbol="ETH/USDT",
            candles_by_timeframe=mock_candles, # re-use
            strategy_id="DUOAI_Strategy",
            trade_context=mock_trade_context_neutral_profit, # use new context
            current_bias=0.5,
            current_confidence=0.5,
            mode='backtest',
            prompt_type='comprehensive_analysis',
            pattern_data=None
        )
        print("\nResultaat van reflectiecyclus (Neutral Sentiment / Profit):")
        print(json.dumps(result_neutral_profit, indent=2, default=str))


        # print("\n--- Starten van de periodieke reflectie loop (met mock data) ---")
        # await ref_lus.start_reflection_loop(symbols=test_symbols, interval_minutes=1)

    asyncio.run(run_test_reflection_loop())

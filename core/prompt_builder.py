# core/prompt_builder.py
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

# Import AI modules that PromptBuilder will use
from .cnn_patterns import CNNPatterns
from .grok_sentiment_fetcher import GrokSentimentFetcher
# from .params_manager import ParamsManager # Only if PromptBuilder itself needs direct param access

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Set level in main application or test script

class PromptBuilder:
    def __init__(self):
        # self.params_manager = ParamsManager() # If PromptBuilder needs its own ParamManager
        self.cnn_patterns = CNNPatterns() # CNNPatterns instantiates its own ParamsManager internally
        try:
            self.sentiment_fetcher = GrokSentimentFetcher()
            logger.info("GrokSentimentFetcher geïnitialiseerd in PromptBuilder.")
        except ValueError as e:
            logger.error(f"Initialisatie GrokSentimentFetcher in PromptBuilder mislukt: {e}. Sentiment features zijn niet beschikbaar.")
            self.sentiment_fetcher = None

        logger.info("PromptBuilder geïnitialiseerd met CNNPatterns en (mogelijk) GrokSentimentFetcher.")

    async def generate_prompt_with_data(
        self,
        candles_by_timeframe: Dict[str, pd.DataFrame],
        symbol: str,
        prompt_type: str,
        current_bias: float = 0.5,
        current_confidence: float = 0.5,
        trade_context: Optional[Dict[str, Any]] = None
    ) -> str:
        logger.debug(f"Genereren prompt type '{prompt_type}' voor {symbol} met {len(candles_by_timeframe)} timeframes.")

        prompt_parts = [
            f"--- Basis Context ---",
            f"Symbool: {symbol}",
            f"Huidige AI Bias: {current_bias:.2f}",
            f"Huidige AI Confidence: {current_confidence:.2f}",
            f"Prompt Type: {prompt_type}",
            f"Datum/Tijd: {datetime.now().isoformat()}"
        ]

        # --- Marktstructuur & Patroon Analyse (CNNPatterns) ---
        prompt_parts.append("\n--- Marktstructuur & Patroon Analyse ---")
        try:
            all_detected_patterns = await self.cnn_patterns.detect_patterns_multi_timeframe(candles_by_timeframe, symbol)
            if all_detected_patterns:
                summary_parts = []
                if "overall_summary" in all_detected_patterns and all_detected_patterns["overall_summary"]:
                    summary_parts.append(f"Algemene Samenvatting: {all_detected_patterns['overall_summary']}")

                if "key_observations" in all_detected_patterns and all_detected_patterns["key_observations"]:
                    obs_str = "; ".join(all_detected_patterns['key_observations'])
                    summary_parts.append(f"Kernobservaties: {obs_str}")

                if "emerging_patterns" in all_detected_patterns and all_detected_patterns["emerging_patterns"]:
                     patterns_str = "; ".join(all_detected_patterns["emerging_patterns"])
                     summary_parts.append(f"Opkomende Patronen: {patterns_str}")

                if "contextual_info" in all_detected_patterns and all_detected_patterns["contextual_info"]:
                    context_info = all_detected_patterns["contextual_info"]
                    summary_parts.append(f"Algemene Context: Trend={context_info.get('trend', 'N/A')}, Volume Profiel={context_info.get('volume_profile', 'N/A')}, Volatiliteit={context_info.get('volatility', 'N/A')}")

                prompt_parts.append(". ".join(summary_parts))

                if "timeframe_details" in all_detected_patterns and all_detected_patterns["timeframe_details"]:
                    prompt_parts.append("\nDetails per Timeframe:")
                    for tf, details in all_detected_patterns["timeframe_details"].items():
                        patterns_str = ", ".join(details.get("patterns", ["geen"]))
                        prediction_str = f"Voorspelling: {details.get('prediction_label')} (Conf: {details.get('prediction_confidence', 0):.2f})" if "prediction_label" in details else "Geen voorspelling"
                        prompt_parts.append(f"  > {tf}: Patronen=[{patterns_str}], {prediction_str}, Marktconditie: {details.get('market_condition', 'N/A')}")
            else:
                prompt_parts.append("Geen patroon data beschikbaar via CNNPatterns.")
        except Exception as e:
            logger.error(f"Fout bij detecteren/verwerken van patronen via CNNPatterns: {e}", exc_info=True)
            prompt_parts.append("Fout bij patroon analyse.")

        # --- Sentiment Analyse (GrokSentimentFetcher) ---
        prompt_parts.append("\n--- Sentiment Analyse ---")
        if self.sentiment_fetcher:
            try:
                base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
                sentiment_data_list = await self.sentiment_fetcher.fetch_live_search_data(base_symbol)
                if sentiment_data_list:
                    prompt_parts.append(f"Recent sentiment voor {base_symbol} (max 3 items):")
                    for i, item in enumerate(sentiment_data_list[:3]): # Limit to 3 items for brevity in prompt
                        title = item.get('title', 'N/A')
                        source = item.get('source', 'N/A')
                        sentiment_score_val = item.get('sentiment_score', 'N/A')
                        sentiment_label_val = item.get('sentiment_label', 'N/A')
                        prompt_parts.append(f"  {i+1}. [{sentiment_label_val} ({sentiment_score_val})] {title} (Bron: {source})")
                else:
                    prompt_parts.append(f"Geen recente sentiment data gevonden voor {base_symbol}.")
            except Exception as e:
                logger.error(f"Fout bij ophalen sentiment in PromptBuilder: {e}", exc_info=True)
                prompt_parts.append(f"Sentiment data kon niet worden opgehaald voor {base_symbol} vanwege een fout.")
        else:
            prompt_parts.append("Sentiment Analyse Service: Niet beschikbaar.")

        # --- Trade Context ---
        if prompt_type == 'tradeReflection' and trade_context:
            prompt_parts.append("\n--- Trade Reflectie Context ---")
            for key, value in trade_context.items():
                prompt_parts.append(f" - {key.replace('_', ' ').capitalize()}: {value}")

        # --- AI Instructie ---
        prompt_parts.append("\n--- Instructie voor AI ---")
        if prompt_type == 'marketAnalysis':
            prompt_parts.append("Analyseer de huidige marktstructuur, patronen en sentiment. Wat is je algemene sentiment (bullish/bearish/neutraal)? Identificeer belangrijke support/resistance zones. Welke patronen zijn dominant en op welke timeframes? Wat is je verwachting voor de komende paar candles op de kortere timeframes? Geef een confidence score (0.0-1.0) en bias score (0.0-1.0) voor je algehele analyse.")
        elif prompt_type == 'tradeReflection':
            prompt_parts.append("Reflecteer op de gesloten trade binnen de gegeven markt- en sentimentcontext. Wat ging goed/fout? Waren de entry/exit punten optimaal gegeven de data? Welke lessen zijn geleerd? Moet de strategie (parameters, entry/exit logica) worden aangepast? Pas de confidence en bias scores aan indien nodig op basis van deze reflectie.")
        else:
            prompt_parts.append("Geef je algemene analyse, conclusies en aanbevelingen op basis van alle verstrekte informatie.")

        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"Gegenereerde prompt voor {symbol} ({prompt_type}):\n{final_prompt[:1000]}...") # Log more for debugging
        return final_prompt

    # Deprecated methods, kept for potential internal testing or phased removal
    async def generate_market_summary_prompt(self, data: Dict[str, Any]):
        logger.warning("generate_market_summary_prompt is verouderd; gebruik generate_prompt_with_data.")
        now_dt = datetime.now()
        mock_candles = { "1h": pd.DataFrame({'close': [float(data.get("price", 100.0))],
                                             'open': [float(data.get("price", 100.0))-1],
                                             'high': [float(data.get("price", 100.0))+1],
                                             'low': [float(data.get("price", 100.0))-2],
                                             'volume': [float(data.get("volume", 1000.0))]},
                                             index=[pd.to_datetime(now_dt)])}
        return await self.generate_prompt_with_data(mock_candles, data.get("symbol", "UNKNOWN_SYMBOL"), "marketAnalysis")

    async def generate_reflection_prompt(self, data: Dict[str, Any]):
        logger.warning("generate_reflection_prompt is verouderd; gebruik generate_prompt_with_data.")
        now_dt = datetime.now()
        mock_candles = { "1h": pd.DataFrame({'close': [float(data.get("entry_price", 100.0))],
                                             'open': [float(data.get("entry_price", 100.0))-1],
                                             'high': [float(data.get("entry_price", 100.0))+1],
                                             'low': [float(data.get("entry_price", 100.0))-2],
                                             'volume': [1000.0]},
                                             index=[pd.to_datetime(now_dt)])}
        return await self.generate_prompt_with_data(mock_candles, data.get("symbol", "UNKNOWN_SYMBOL"), "tradeReflection", trade_context=data)

async def test_mock_prompt_builder():
    import sys
    # Ensure logger for this test is at DEBUG level to see all prompt parts
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    # Reconfigure existing logger for test, or specific logger for PromptBuilder
    logging.getLogger('core.prompt_builder').setLevel(logging.DEBUG)


    builder = PromptBuilder()

    # Mock CNNPatterns methods
    async def mock_detect_patterns_multi_timeframe(candles_by_timeframe, symbol):
        logger.info(f"[MOCK CNNPatterns] Detecting patterns for {symbol} on {list(candles_by_timeframe.keys())}")
        return {
            "overall_summary": "Overall market shows mixed signals with some bullish indications on lower timeframes.",
            "key_observations": ["Volume spike on 1h.", "RSI divergence on 5m."],
            "emerging_patterns": ["Possible bull flag forming on 1h timeframe."],
            "contextual_info": {"trend": "sideways_uptrend", "volume_profile": "P-shape", "volatility": "medium"},
            "timeframe_details": {
                "5m": {"patterns": ["rsi_divergence"], "prediction_label": "bullish", "prediction_confidence": 0.6, "market_condition": "choppy"},
                "1h": {"patterns": ["bull_flag_developing", "volume_spike"], "prediction_label": "potential_bullish_continuation", "prediction_confidence": 0.75, "market_condition": "consolidation_after_upmove"}
            }
        }
    builder.cnn_patterns.detect_patterns_multi_timeframe = mock_detect_patterns_multi_timeframe

    # Mock GrokSentimentFetcher methods
    if builder.sentiment_fetcher:
        async def mock_fetch_live_search_data(base_symbol):
            logger.info(f"[MOCK GrokSentimentFetcher] Fetching sentiment for {base_symbol}")
            return [
                {"title": "Big news for ETH!", "source": "CryptoNews", "content": "ETH to the moon after new partnership announced with tech giant. Market reacts positively.", "sentiment_score": 0.8, "sentiment_label": "very positive"},
                {"title": "ETH price drops slightly", "source": "CoinDesk", "content": "Minor correction in ETH price today, analysts not worried about long term.", "sentiment_score": -0.2, "sentiment_label": "slightly negative"}
            ]
        builder.sentiment_fetcher.fetch_live_search_data = mock_fetch_live_search_data
    else:
        logger.warning("[Test PromptBuilder] Sentiment fetcher was not initialized, sentiment will be unavailable in test.")

    now_dt = datetime.now()
    mock_data_5m = {'date': pd.to_datetime([now_dt - timedelta(minutes=i*5) for i in range(10, 0, -1)]), # Ensure recent first
                    'open': np.random.rand(10) * 10 + 100, 'high': np.random.rand(10) * 5 + 110,
                    'low': np.random.rand(10) * 5 + 95, 'close': np.random.rand(10) * 10 + 100,
                    'volume': np.random.rand(10) * 1000 + 500}
    df_5m = pd.DataFrame(mock_data_5m).set_index('date')

    mock_data_1h = {'date': pd.to_datetime([now_dt - timedelta(hours=i) for i in range(10, 0, -1)]),
                    'open': np.random.rand(10) * 10 + 100, 'high': np.random.rand(10) * 5 + 110,
                    'low': np.random.rand(10) * 5 + 95, 'close': np.random.rand(10) * 10 + 100,
                    'volume': np.random.rand(10) * 2000 + 1000}
    df_1h = pd.DataFrame(mock_data_1h).set_index('date')

    candles_dict = {"5m": df_5m, "1h": df_1h}

    print("\n--- Test Market Analysis Prompt (with Mocks) ---")
    market_analysis_prompt = await builder.generate_prompt_with_data(
        candles_by_timeframe=candles_dict, symbol="ETH/EUR", prompt_type="marketAnalysis",
        current_bias=0.55, current_confidence=0.65
    )
    print(market_analysis_prompt)

    print("\n--- Test Trade Reflection Prompt (with Mocks) ---")
    trade_reflection_prompt = await builder.generate_prompt_with_data(
        candles_by_timeframe=candles_dict, symbol="ETH/EUR", prompt_type="tradeReflection",
        trade_context={"profit_pct": -0.05, "exit_reason": "stop_loss", "duration_minutes": 60},
        current_bias=0.4, current_confidence=0.5
    )
    print(trade_reflection_prompt)

if __name__ == '__main__':
    # Ensure top-level logging is also DEBUG if we want to see everything from test run
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_mock_prompt_builder())

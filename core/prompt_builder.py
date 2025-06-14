import asyncio
import logging
import json
from core.grok_sentiment_fetcher import GrokSentimentFetcher
from unittest.mock import AsyncMock, MagicMock # Added for testing
import pandas as pd

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self, grok_fetcher_instance=None): # Allow injecting a fetcher
        logger.info("PromptBuilder initialized.")
        if grok_fetcher_instance:
            self.grok_sentiment_fetcher = grok_fetcher_instance
        else:
            try:
                self.grok_sentiment_fetcher = GrokSentimentFetcher()
            except ValueError as e: # Typically API key missing
                logger.warning(f"Failed to initialize default GrokSentimentFetcher (ValueError): {e}. Social sentiment will not be fetched.")
                self.grok_sentiment_fetcher = None
            except ImportError: # If GrokSentimentFetcher class itself is not found
                logger.warning("GrokSentimentFetcher class not found by PromptBuilder. Social sentiment will not be fetched.")
                self.grok_sentiment_fetcher = None
            except Exception as e: # Catch any other unexpected errors during init
                logger.error(f"An unexpected error occurred during GrokSentimentFetcher initialization: {e}")
                self.grok_sentiment_fetcher = None


    async def generate_prompt_with_data(
        self,
        candles_by_timeframe: dict,
        symbol: str,
        prompt_type: str,
        current_bias: float = 0.5,
        current_confidence: float = 0.5,
        additional_context: dict = None
    ) -> str:
        """
        Generates a detailed prompt for AI analysis based on provided market data and context.
        """
        final_prompt_parts = []
        all_timeframe_indicators = {}

        # --- Social Sentiment Data Handling (Prioritize additional_context) ---
        grok_data = [] # Default to empty list
        source_of_grok_data = "not_applicable" # For logging
        prompt_types_for_grok_fetch = ['marketAnalysis', 'riskManagement', 'exitSignal'] # Configured prompt types for fetching

        # Check additional_context first
        passed_sentiment_data = additional_context.get('social_sentiment_grok') if additional_context else None

        if passed_sentiment_data and isinstance(passed_sentiment_data, list) and len(passed_sentiment_data) > 0:
            grok_data = passed_sentiment_data
            source_of_grok_data = "additional_context"
            logger.info(f"Using {len(grok_data)} social sentiment items provided via additional_context for {symbol}.")
        elif self.grok_sentiment_fetcher:
            should_fetch_grok_data_for_prompt_type = prompt_type in prompt_types_for_grok_fetch
            if should_fetch_grok_data_for_prompt_type:
                logger.info(f"Attempting to fetch Grok sentiment for {symbol} (prompt: {prompt_type}) as not provided or empty in additional_context.")
                try:
                    grok_data = await self.grok_sentiment_fetcher.fetch_live_search_data(
                        symbol=symbol,
                        perform_fetch=True
                    )
                    source_of_grok_data = "fetched"
                    logger.info(f"Successfully fetched {len(grok_data)} Grok items for {symbol}.")
                except Exception as e:
                    logger.error(f"Error fetching Grok data in PromptBuilder for {symbol}: {e}")
                    source_of_grok_data = "fetch_failed"
                    # grok_data remains []
            else:
                source_of_grok_data = "fetch_skipped_prompt_type"
                logger.info(f"Skipping Grok sentiment fetch for {symbol} (prompt_type: {prompt_type} not configured for fetch, and no data passed).")
        else:
            source_of_grok_data = "fetcher_unavailable"
            logger.warning(f"GrokSentimentFetcher not available and no sentiment data passed, skipping Grok fetch for {symbol}.")
        # --- End Social Sentiment Data Handling ---

        # Common header
        final_prompt_parts.append(f"Symbol: {symbol}")
        final_prompt_parts.append(f"Current AI Bias: {current_bias:.2f}, Current AI Confidence: {current_confidence:.2f}")

        if prompt_type == 'marketAnalysis':
            final_prompt_parts.append("Prompt Type: Market Analysis for Potential Entry")
            final_prompt_parts.append(
                "Analyze the following market data, including any provided social media sentiment. "
                "Provide your trade intention (LONG, SHORT, HOLD), confidence level (0.0-1.0), and perceived bias (0.0-1.0). "
                "Explicitly analyze the provided social media sentiment. Detail how this sentiment influences your interpretation of technical indicators, chart patterns, and your overall trading decision. "
                "For example: 'Based on the technicals I see X (e.g., bullish RSI divergence), the detected chart pattern is Z (e.g., a double bottom), and the positive social sentiment further supports this by indicating Y (e.g., strong community backing for an upcoming feature), leading to a LONG decision with high confidence...'"
            )
            # Add more specific instructions for market analysis if needed

        elif prompt_type == 'riskManagement':
            final_prompt_parts.append("Prompt Type: Risk Management Assessment")
            final_prompt_parts.append(
                "Based on the current market data and the provided trade context (if any), "
                "assess the risk. If you recommend a specific stop-loss percentage, "
                "please provide it ONLY in a JSON object within your response, formatted as: "
                "`{\"recommended_sl_percentage\": X.Y}` where X.Y is the percentage (e.g., 2.5 for 2.5%). "
                "If you do not recommend a specific SL percentage, do not include this JSON object. "
                "Your textual explanation for your reasoning should precede this JSON object if present. "
                "Also, state your confidence in this SL recommendation."
            )
            if additional_context:
                final_prompt_parts.append("\nAdditional Trade Context:")
                for key, value in additional_context.items():
                    final_prompt_parts.append(f"- {key.replace('_', ' ').capitalize()}: {value}")

        elif prompt_type == 'exitSignal':
            final_prompt_parts.append("Prompt Type: Trade Exit Signal Evaluation")
            # Add specific instructions for exit signal
            final_prompt_parts.append("Evaluate if the current trade should be exited. Provide your reasoning and confidence.")


        # Data section (simplified - a real implementation would format candles)
        final_prompt_parts.append("\nMarket Data:")
        for tf, df in candles_by_timeframe.items():
            if not df.empty:
                final_prompt_parts.append(f"\n--- Timeframe: {tf} ---")
                final_prompt_parts.append(f"Latest {min(5, len(df))} candles (condensed):")
                # In a real version, select key columns and format them nicely
                final_prompt_parts.append(df.tail(min(5, len(df))).to_string(columns=['open', 'high', 'low', 'close', 'volume']))
            else:
                final_prompt_parts.append(f"\n--- Timeframe: {tf} --- (No data provided)")

        # Technical Indicators Section
        primary_indicators = [ # These are typically always checked if present
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'bb_upperband', 'bb_middleband', 'bb_lowerband',
            # Specific EMAs like ema_short, ema_long are explicitly listed
            # if they have fixed meanings in the strategy.
            # Dynamic EMAs (ema_*) will be caught by the loop below.
        ]
        all_timeframe_candlestick_patterns = {}

        for tf, df in candles_by_timeframe.items():
            if not df.empty:
                latest_row = df.iloc[-1]
                indicators_for_tf = {}
                candlestick_patterns_for_tf = []

                # Populate primary indicators
                for indicator_col in primary_indicators:
                    if indicator_col in df.columns and pd.notna(latest_row[indicator_col]):
                        indicators_for_tf[indicator_col] = latest_row[indicator_col]

                # Dynamically find all ema_* and volume_mean_* columns
                for col_name in df.columns:
                    if (col_name.startswith('ema_') or col_name.startswith('volume_mean_')) and \
                       pd.notna(latest_row[col_name]):
                        indicators_for_tf[col_name] = latest_row[col_name]

                    # Candlestick patterns
                    if col_name.startswith('CDL') and latest_row[col_name] != 0 and pd.notna(latest_row[col_name]):
                        candlestick_patterns_for_tf.append(col_name)


                if indicators_for_tf:
                    all_timeframe_indicators[tf] = indicators_for_tf

                if candlestick_patterns_for_tf:
                    all_timeframe_candlestick_patterns[tf] = candlestick_patterns_for_tf

        if all_timeframe_indicators:
            final_prompt_parts.append("\nLatest Technical Indicators:")
            final_prompt_parts.append(json.dumps(all_timeframe_indicators, indent=2, default=str))

        if all_timeframe_candlestick_patterns:
            final_prompt_parts.append("\nDetected Candlestick Patterns:")
            final_prompt_parts.append(json.dumps(all_timeframe_candlestick_patterns, indent=2, default=str))

        # CNN Patterns Section
        if additional_context:
            cnn_predictions = additional_context.get('pattern_data', {}).get('cnn_predictions')
            if cnn_predictions: # If it's a non-empty dict
                final_prompt_parts.append("\nDetected CNN Patterns:")
                final_prompt_parts.append(json.dumps(cnn_predictions, indent=2, default=str))

            # Social Sentiment Data Section from Grok
        # Determine if we should attempt to show the sentiment section
        # This is true if data was passed, or if it's a fetch-configured prompt type and the fetcher is available.
        attempted_to_include_sentiment = (source_of_grok_data == "additional_context") or \
                                         (self.grok_sentiment_fetcher and prompt_type in prompt_types_for_grok_fetch)

        if attempted_to_include_sentiment:
            final_prompt_parts.append("\nSocial Sentiment/News (Grok):")
            if grok_data: # If data is present (either passed or fetched successfully)
                for i, item in enumerate(grok_data):
                    source = item.get('source', 'N/A')
                    text = item.get('text', 'N/A')
                    sentiment = item.get('sentiment', 'N/A')
                    text_snippet = (text[:147] + "...") if len(text) > 150 else text
                    final_prompt_parts.append(f"  --- Item {i+1} ---")
                    final_prompt_parts.append(f"  Source: {source}")
                    final_prompt_parts.append(f"  Sentiment: {sentiment}")
                    final_prompt_parts.append(f"  Text: {text_snippet}")
            else: # Attempted to include (passed or fetched for this prompt type), but grok_data ended up empty.
                final_prompt_parts.append("  No recent social sentiment/news items found from Grok.")
        # If not attempted_to_include_sentiment (e.g., fetcher unavailable AND no data passed,
        # OR prompt type not relevant for fetching AND no data passed), this whole section is skipped.

        final_prompt = "\n".join(final_prompt_parts)
        # logger.debug(f"Generated prompt for type '{prompt_type}' for {symbol}:\n{final_prompt}")
        return final_prompt


async def test_prompt_builder():
    # --- Mock setup ---
    mock_grok_fetcher = MagicMock() # This is the instance of the fetcher
    mock_grok_fetcher.fetch_live_search_data = AsyncMock() # fetch_live_search_data is an async method

    # Instantiate PromptBuilder with the mock fetcher
    builder = PromptBuilder(grok_fetcher_instance=mock_grok_fetcher)

    # --- Test Data ---
    mock_candles = {
        '5m': pd.DataFrame({
            'open': [10, 11, 12, 11.5, 12.5],
            'high': [10.5, 11.5, 12.5, 12, 13],
            'low': [9.5, 10.5, 11.5, 11, 12],
            'close': [11, 12, 11.5, 12.5, 12.8],
            'volume': [100, 120, 110, 130, 90],
            'rsi': [50, 55, 60, 58, 62.1],
            'macd': [0.1, 0.12, 0.15, 0.13, 0.16],
            'macdsignal': [0.09, 0.1, 0.11, 0.12, 0.13],
            'macdhist': [0.01, 0.02, 0.04, 0.01, 0.03],
            'bb_lowerband': [9, 10, 11, 10.5, 11.5],
            'bb_middleband': [10, 11, 12, 11.5, 12.5],
            'bb_upperband': [11, 12, 13, 12.5, 13.5],
            'ema_short': [10.8, 11.2, 11.8, 11.6, 12.2], # Explicitly named
            'ema_long': [10.5, 10.8, 11.1, 11.3, 11.7],  # Explicitly named
            'ema_20': [10.7, 11.1, 11.7, 11.5, 12.1],    # Dynamic
            'ema_100': [10.2, 10.5, 10.8, 11.0, 11.4],   # Dynamic
            'volume_mean_10': [110, 115, 112, 118, 110.5],
            'volume_mean_30': [105, 110, 115, 112, 108.0],
            'CDLMORNINGSTAR': [0, 0, 0, 0, 100], # Active pattern
            'CDLDOJI': [0, 0, 0, 0, 0],        # Inactive pattern
            'CDLHAMMER': [0, 100, 0, 0, 0]      # Pattern not on latest candle
        }),
        '1h': pd.DataFrame({
            'open': [1200, 1210, 1205],
            'high': [1250, 1220, 1215],
            'low': [1190, 1200, 1200],
            'close': [1210, 1205, 1212.7],
            'volume': [1000, 1200, 1100],
            'rsi': [60.3, 55.2, 58.9],
            'ema_short': [1208, 1206, 1209.5], # Explicitly named
            'ema_50': [1200, 1202, 1205.0],    # Dynamic
            'CDLENGULFING': [0, 0, -100],      # Active pattern
            'CDLSHOOTINGSTAR': [0,0,0]
        })
    }

    mock_additional_context_no_sentiment = { # Renamed for clarity
        'pattern_data': {
            'cnn_predictions': {
                '5m_bullFlag_score': 0.92,
                '1h_bearTrap_score': 0.78,
                '5m_randomPattern_score': 0.3
            }
        },
        "current_trade": {"pair": "BTC/USDT", "profit_pct": 5.2, "open_reason": "EMA cross"},
    }

    mock_additional_context_with_sentiment = {
        **mock_additional_context_no_sentiment, # Inherits CNN, trade context
        "social_sentiment_grok": [
            {"source": "ContextFeed", "text": "Positive sentiment from context for BTC.", "sentiment": "positive"},
            {"source": "ContextSite", "text": "Neutral sentiment from context for BTC.", "sentiment": "neutral"},
        ]
    }


    # --- Test Case 1 (New): Data provided via additional_context for a fetching prompt type ---
    logger.info("--- Test Case 1: Market Analysis with social data from additional_context ---")
    prompt_ma_passed_data = await builder.generate_prompt_with_data(
        mock_candles,
        "BTC/USDT",
        "marketAnalysis", # This prompt type would normally fetch
        additional_context=mock_additional_context_with_sentiment
    )
    logger.info(f"--- Market Analysis Prompt (passed data) ---\n{prompt_ma_passed_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_not_called() # Should NOT fetch
    assert "Social Sentiment/News (Grok):" in prompt_ma_passed_data
    assert "Source: ContextFeed" in prompt_ma_passed_data # Data from additional_context
    assert "Sentiment: positive" in prompt_ma_passed_data
    assert "Text: Positive sentiment from context for BTC." in prompt_ma_passed_data
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Case 1 Passed.")


    # --- Test Case 2 (Adjusted): Market Analysis with social data from fetcher (no data in context) ---
    logger.info("--- Test Case 2: Market Analysis with social data from fetcher (no data in context) ---")
    mock_grok_fetcher.fetch_live_search_data.return_value = [
        {"source": "MockX", "text": "Mock positive news for BTC.", "sentiment": "positive"},
        {"source": "MockSite", "text": "Mock neutral news for BTC.", "sentiment": "neutral"},
    ]
    prompt_ma_fetched_data = await builder.generate_prompt_with_data(
        mock_candles,
        "BTC/USDT",
        "marketAnalysis",
        additional_context=mock_additional_context_no_sentiment # No sentiment here
    )
    logger.info(f"--- Market Analysis Prompt (fetched data) ---\n{prompt_ma_fetched_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_called_once_with(symbol="BTC/USDT", perform_fetch=True)
    assert "Social Sentiment/News (Grok):" in prompt_ma_fetched_data
    assert "Source: MockX" in prompt_ma_fetched_data # Data from fetcher
    assert "Sentiment: positive" in prompt_ma_fetched_data
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Case 2 Passed.")


    # --- Existing Assertions for Technical Indicators, Candlestick, CNN (can use prompt_ma_fetched_data) ---
    assert "Latest Technical Indicators:" in prompt_ma_fetched_data
    assert '"5m": {' in prompt_ma_fetched_data
    assert '"rsi": 62.1' in prompt_ma_fetched_data
    # ... (other existing TI assertions remain the same, ensure they use a valid prompt like prompt_ma_fetched_data)
    assert "Detected Candlestick Patterns:" in prompt_ma_fetched_data

    # Helper function to extract JSON blocks robustly (remains the same)
    def extract_json_block(prompt_text, start_marker, end_marker=None):
        try:
            after_start_marker = prompt_text.split(start_marker, 1)[1]
            # If an end_marker is provided and found, split by it
            if end_marker and end_marker in after_start_marker:
                json_str = after_start_marker.split(end_marker, 1)[0].strip()
            else:
                # Otherwise, assume the JSON string goes to the end of the text (or before next section)
                json_str = after_start_marker.strip()
            return json.loads(json_str)
        except IndexError: # start_marker not found
            logger.error(f"Start marker '{start_marker}' not found in prompt.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError for block starting with '{start_marker}': {e}")
            logger.error(f"Problematic JSON string part: {json_str[:200]}...")
            return None

    candlestick_data = extract_json_block(prompt_ma_with_data, "Detected Candlestick Patterns:\n", "\nDetected CNN Patterns:")
    assert candlestick_data is not None, "Failed to parse candlestick data"
    assert "5m" in candlestick_data
    assert "CDLMORNINGSTAR" in candlestick_data["5m"]
    assert "CDLDOJI" not in candlestick_data["5m"]
    assert "CDLHAMMER" not in candlestick_data["5m"]
    assert "1h" in candlestick_data
    assert "CDLENGULFING" in candlestick_data["1h"]
    assert "CDLSHOOTINGSTAR" not in candlestick_data["1h"]

    # Assertions for CNN Patterns
    assert "Detected CNN Patterns:" in prompt_ma
    # Assuming CNN patterns section is the last one with JSON, or followed by non-JSON content.
    # If another section could follow, its start marker would be the end_marker here.
    # The end marker for CNN data is now "\nSocial Sentiment/News (Grok):" if Grok data is present.
    cnn_data = extract_json_block(prompt_ma_fetched_data, "Detected CNN Patterns:\n", "\nSocial Sentiment/News (Grok):")
    assert cnn_data is not None, "Failed to parse CNN data"
    assert cnn_data['5m_bullFlag_score'] == 0.92
    assert cnn_data['1h_bearTrap_score'] == 0.78 # Ensure this uses prompt_ma_fetched_data
    assert cnn_data['5m_randomPattern_score'] == 0.3 # Ensure this uses prompt_ma_fetched_data
    logger.info("Existing assertions for TI, Candlesticks, CNN passed.")


    # --- Test Case 3 (Adjusted): Market Analysis with fetcher returning empty list (no data in context) ---
    logger.info("--- Test Case 3: Market Analysis with fetcher returning empty list (no data in context) ---")
    mock_grok_fetcher.fetch_live_search_data.return_value = []
    prompt_ma_empty_fetched_data = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "marketAnalysis", additional_context=mock_additional_context_no_sentiment
    )
    logger.info(f"--- Market Analysis Prompt (empty fetched data) ---\n{prompt_ma_empty_fetched_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_called_once_with(symbol="ETH/USDT", perform_fetch=True)
    assert "Social Sentiment/News (Grok):" in prompt_ma_empty_fetched_data
    assert "No recent social sentiment/news items found from Grok." in prompt_ma_empty_fetched_data
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Case 3 Passed.")

    # --- Test Case 4 (Adjusted): Prompt type where fetch is not performed (no data in context) ---
    logger.info("--- Test Case 4: Prompt type where fetch is not performed (no data in context) ---")
    mock_grok_fetcher.fetch_live_search_data.return_value = [{"source": "ShouldNotAppear", "text": "Data", "sentiment": "none"}] # Should not be called
    prompt_status_update_no_context_data = await builder.generate_prompt_with_data(
        mock_candles, "BTC/USDT", "statusUpdate", # statusUpdate is not in prompt_types_for_grok_fetch
        additional_context=mock_additional_context_no_sentiment
    )
    logger.info(f"--- Status Update Prompt (no context data, no fetch) ---\n{prompt_status_update_no_context_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_not_called()
    assert "Social Sentiment/News (Grok):" not in prompt_status_update_no_context_data # Section should be absent
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Case 4 Passed.")

    # --- Test Case 5 (New): Data provided via additional_context for a normally non-fetching prompt type ---
    logger.info("--- Test Case 5: Data provided via additional_context for a non-fetching prompt type ---")
    prompt_status_update_passed_data = await builder.generate_prompt_with_data(
        mock_candles,
        "BTC/USDT",
        "statusUpdate", # This prompt type would normally NOT fetch
        additional_context=mock_additional_context_with_sentiment # But data is passed
    )
    logger.info(f"--- Status Update Prompt (passed data) ---\n{prompt_status_update_passed_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_not_called() # Still should NOT fetch
    assert "Social Sentiment/News (Grok):" in prompt_status_update_passed_data # Section IS present
    assert "Source: ContextFeed" in prompt_status_update_passed_data # Data from additional_context
    assert "Text: Positive sentiment from context for BTC." in prompt_status_update_passed_data
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Case 5 Passed.")


    # --- Test Case 6 (Adjusted from old Test Case 4): GrokSentimentFetcher is None (simulating initialization failure) ---
    logger.info("--- Test Case 6: GrokSentimentFetcher is None (no data in context) ---")
    builder_no_fetcher = PromptBuilder(grok_fetcher_instance=None)
    # This will try to init GrokSentimentFetcher, which might succeed or fail depending on env (e.g. GROK_API_KEY)
    # If it fails, self.grok_sentiment_fetcher will be None internally.
    # If it succeeds, this test case won't be a true test of it being None unless GROK_API_KEY is unset.
    # For robust local testing, one might temporarily unset GROK_API_KEY for this specific builder instance.
    # However, the code is designed so that if self.grok_sentiment_fetcher is None, it skips fetching.

    # To ensure self.grok_sentiment_fetcher is None for the test, we can check its state.
    # If builder_no_fetcher.grok_sentiment_fetcher is not None here, it means default init worked.
    # This test is more about the PromptBuilder's behavior when its fetcher attribute *is* None.
    # Let's assume the internal try-except in PromptBuilder sets it to None if GrokFetcher init fails.

    # If GROK_API_KEY is actually set in the test environment, GrokSentimentFetcher() might get initialized.
    # To truly test the "None" path without manipulating environment variables during the test run,
    # we rely on the fact that if grok_fetcher_instance=None is passed, AND the internal
    # instantiation fails (e.g. due to missing key or ImportError), it becomes None.
    # We can force it to be None for this specific test instance for reliable testing:
    builder_no_fetcher.grok_sentiment_fetcher = None # Force the instance's fetcher to be None

    prompt_ma_no_fetcher_no_context_data = await builder_no_fetcher.generate_prompt_with_data(
        mock_candles, "ADA/USDT", "marketAnalysis",
        additional_context=mock_additional_context_no_sentiment # No data in context
    )
    logger.info(f"--- Market Analysis Prompt (no fetcher, no context data) ---\n{prompt_ma_no_fetcher_no_context_data}\n")
    assert "Social Sentiment/News (Grok):" not in prompt_ma_no_fetcher_no_context_data # Section should be absent
    logger.info("Test Case 6 Passed.")

    # --- Test Case 7 (New): GrokSentimentFetcher is None, but data IS provided via additional_context ---
    logger.info("--- Test Case 7: GrokSentimentFetcher is None, but data IS provided via additional_context ---")
    # builder_no_fetcher still has its grok_sentiment_fetcher as None
    prompt_ma_no_fetcher_with_context_data = await builder_no_fetcher.generate_prompt_with_data(
        mock_candles, "ADA/USDT", "marketAnalysis",
        additional_context=mock_additional_context_with_sentiment # Data IS in context
    )
    logger.info(f"--- Market Analysis Prompt (no fetcher, but with context data) ---\n{prompt_ma_no_fetcher_with_context_data}\n")
    assert "Social Sentiment/News (Grok):" in prompt_ma_no_fetcher_with_context_data # Section IS present
    assert "Source: ContextFeed" in prompt_ma_no_fetcher_with_context_data # Data from additional_context
    logger.info("Test Case 7 Passed.")


    # --- Test Risk Management prompt (adjusted to fit new structure) ---
    logger.info("--- Test Risk Management: Fetching when no data in context ---")
    mock_grok_fetcher.fetch_live_search_data.return_value = [
        {"source": "RiskMock", "text": "Critical risk warning for ETH.", "sentiment": "negative"},
    ]
    # Use the original 'builder' which has the mock_grok_fetcher
    # mock_additional_context_no_sentiment already has necessary CNN/trade data
    prompt_rm_fetched_data = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "riskManagement",
        current_bias=0.6, current_confidence=0.75,
        additional_context=mock_additional_context_no_sentiment
    )
    logger.info(f"--- Risk Management Prompt (fetched data) ---\n{prompt_rm_fetched_data}\n")
    mock_grok_fetcher.fetch_live_search_data.assert_called_once_with(symbol="ETH/USDT", perform_fetch=True)
    assert "Social Sentiment/News (Grok):" in prompt_rm_fetched_data
    assert "Source: RiskMock" in prompt_rm_fetched_data
    mock_grok_fetcher.fetch_live_search_data.reset_mock()
    logger.info("Test Risk Management (fetching) passed.")

    logger.info("All test cases passed.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_prompt_builder())

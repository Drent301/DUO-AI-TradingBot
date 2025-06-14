import asyncio
import logging
import json
import pandas as pd

logger = logging.getLogger(__name__)

class PromptBuilder:
    def __init__(self):
        # In a real scenario, this might load templates or other resources
        logger.info("PromptBuilder initialized.")

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

        # Common header
        final_prompt_parts.append(f"Symbol: {symbol}")
        final_prompt_parts.append(f"Current AI Bias: {current_bias:.2f}, Current AI Confidence: {current_confidence:.2f}")

        if prompt_type == 'marketAnalysis':
            final_prompt_parts.append("Prompt Type: Market Analysis for Potential Entry")
            final_prompt_parts.append(
                "Analyze the following market data, including any provided social media sentiment. "
                "Provide your trade intention (LONG, SHORT, HOLD), confidence level (0.0-1.0), and perceived bias (0.0-1.0). "
                "Explicitly state how social media sentiment influences your technical analysis and decision-making process. "
                "For example: 'Based on the technicals I see X, and the positive social sentiment further supports/contradicts this by Y...'"
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
            if additional_context and 'social_sentiment_grok' in additional_context:
                social_sentiment_grok_data = additional_context.get('social_sentiment_grok')
                if isinstance(social_sentiment_grok_data, list):
                    final_prompt_parts.append("\nSocial Sentiment/News (Grok):")
                    if not social_sentiment_grok_data:
                        final_prompt_parts.append("  No recent social sentiment/news items found from Grok.")
                    else:
                        for i, item in enumerate(social_sentiment_grok_data):
                            source = item.get('source', 'N/A')
                            text = item.get('text', 'N/A')
                            sentiment = item.get('sentiment', 'N/A')

                            text_snippet = (text[:147] + "...") if len(text) > 150 else text

                            final_prompt_parts.append(f"  --- Item {i+1} ---")
                            final_prompt_parts.append(f"  Source: {source}")
                            final_prompt_parts.append(f"  Sentiment: {sentiment}")
                            final_prompt_parts.append(f"  Text: {text_snippet}")
                elif social_sentiment_grok_data is not None:
                    final_prompt_parts.append("\nSocial Sentiment/News (Grok): (Data provided in unexpected format)")
            # If 'social_sentiment_grok' is not in additional_context or is None, this whole block is skipped.


        final_prompt = "\n".join(final_prompt_parts)
        # logger.debug(f"Generated prompt for type '{prompt_type}' for {symbol}:\n{final_prompt}")
        return final_prompt


async def test_prompt_builder():
    builder = PromptBuilder()
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

    mock_additional_context = {
        'pattern_data': {
            'cnn_predictions': {
                '5m_bullFlag_score': 0.92,
                '1h_bearTrap_score': 0.78,
                '5m_randomPattern_score': 0.3
            }
        },
        "current_trade": {"pair": "BTC/USDT", "profit_pct": 5.2, "open_reason": "EMA cross"},
        "social_sentiment_grok": [
            {
                "source": "CryptoNewsDaily",
                "text": "A new report from top market analysts suggests that Bitcoin (BTC) is poised for a significant rally, potentially reaching $100,000 by the end of the year. This optimistic outlook is fueled by increased institutional adoption and positive regulatory news. However, some caution that market volatility remains a concern.",
                "sentiment": "positive"
            },
            {
                "source": "ETHWorld",
                "text": "The recent Ethereum Developer Conference (DevCon) showcased several promising layer-2 scalability solutions. These advancements aim to address network congestion and high gas fees, paving the way for wider adoption of decentralized applications (dApps).",
                "sentiment": "neutral"
            },
            {
                "source": "AncientScrolls",
                "text": "Old Tweet, No Content.",
                "sentiment": "unknown"
            }
        ]
    }

    prompt_ma = await builder.generate_prompt_with_data(
        mock_candles,
        "BTC/USDT",
        "marketAnalysis",
        additional_context=mock_additional_context
    )
    logger.info(f"--- Market Analysis Prompt ---\n{prompt_ma}\n")

    # Assertions for Technical Indicators
    assert "Latest Technical Indicators:" in prompt_ma
    assert '"5m": {' in prompt_ma
    assert '"rsi": 62.1' in prompt_ma
    assert '"macd": 0.16' in prompt_ma
    assert '"ema_short": 12.2' in prompt_ma # Explicit
    assert '"ema_long": 11.7' in prompt_ma  # Explicit
    assert '"ema_20": 12.1' in prompt_ma    # Dynamic
    assert '"ema_100": 11.4' in prompt_ma   # Dynamic
    assert '"volume_mean_10": 110.5' in prompt_ma
    assert '"volume_mean_30": 108.0' in prompt_ma
    assert '"1h": {' in prompt_ma
    assert '"rsi": 58.9' in prompt_ma
    assert '"ema_short": 1209.5' in prompt_ma # Explicit
    assert '"ema_50": 1205.0' in prompt_ma    # Dynamic
    assert '"macd":' not in prompt_ma.split('"1h": {')[1].split('}')[0] # MACD not in 1h

    # Assertions for Candlestick Patterns
    assert "Detected Candlestick Patterns:" in prompt_ma

    # Helper function to extract JSON blocks robustly
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
            logger.error(f"Problematic JSON string part: {json_str[:200]}...") # Print more context
            return None

    candlestick_data = extract_json_block(prompt_ma, "Detected Candlestick Patterns:\n", "\nDetected CNN Patterns:")
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
    cnn_data = extract_json_block(prompt_ma, "Detected CNN Patterns:\n", "\nSocial Sentiment/News (Grok):")
    assert cnn_data is not None, "Failed to parse CNN data"
    assert cnn_data['5m_bullFlag_score'] == 0.92
    assert cnn_data['1h_bearTrap_score'] == 0.78
    assert cnn_data['5m_randomPattern_score'] == 0.3

    # Assertions for Social Sentiment Data (Grok)
    assert "Social Sentiment/News (Grok):" in prompt_ma
    assert "Source: CryptoNewsDaily" in prompt_ma
    assert "Sentiment: positive" in prompt_ma
    assert "Text: A new report from top market analysts suggests that Bitcoin (BTC) is poised for a significant rally, potentially reaching $100,000 by the end of th..." in prompt_ma
    assert "Source: ETHWorld" in prompt_ma
    assert "Sentiment: neutral" in prompt_ma
    assert "Text: The recent Ethereum Developer Conference (DevCon) showcased several promising layer-2 scalability solutions. These advancements aim to address netw..." in prompt_ma
    assert "Source: AncientScrolls" in prompt_ma
    assert "Sentiment: unknown" in prompt_ma
    assert "Text: Old Tweet, No Content." in prompt_ma
    assert "Timestamp:" not in prompt_ma
    assert "Title:" not in prompt_ma

    # Assertion for the new instruction in marketAnalysis prompt:
    assert "Explicitly state how social media sentiment influences your technical analysis and decision-making process." in prompt_ma
    logger.info("Assertions for Market Analysis prompt passed.")

    # Test riskManagement prompt to ensure context is still handled, including social sentiment
    mock_rm_context = {
        "entry_price": 2000,
        "current_profit_pct": 2.1,
        'pattern_data': {
            'cnn_predictions': {'5m_headAndShoulders_score': 0.85}
        },
        "social_sentiment_grok": [ # Add sentiment to RM test as well, using new key and structure
            {
                "source": "BearishTimes",
                "text": "Several indicators point towards a potential market correction in the short term. Investors are advised to exercise caution.",
                "sentiment": "negative"
            }
        ]
    }
    prompt_rm = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "riskManagement",
        current_bias=0.6, current_confidence=0.75,
        additional_context=mock_rm_context
    )
    logger.info(f"--- Risk Management Prompt ---\n{prompt_rm}\n")
    assert "Prompt Type: Risk Management Assessment" in prompt_rm
    assert "Entry price: 2000" in prompt_rm
    assert "Current profit pct: 2.1" in prompt_rm
    assert "Detected CNN Patterns:" in prompt_rm
    assert '"5m_headAndShoulders_score": 0.85' in prompt_rm
    assert "Social Sentiment/News (Grok):" in prompt_rm # Check sentiment in RM prompt with new key
    assert "Source: BearishTimes" in prompt_rm
    assert "Sentiment: negative" in prompt_rm
    assert "Text: Several indicators point towards a potential market correction in the short term. Investors are advised to exercise caution." in prompt_rm
    logger.info("Assertions for Risk Management prompt passed.")

    # Test with empty social_sentiment_grok list
    empty_sentiment_context = {**mock_additional_context, "social_sentiment_grok": []}
    prompt_empty_sentiment = await builder.generate_prompt_with_data(
        mock_candles, "BTC/USDT", "marketAnalysis", additional_context=empty_sentiment_context
    )
    assert "Social Sentiment/News (Grok):" in prompt_empty_sentiment
    assert "No recent social sentiment/news items found from Grok." in prompt_empty_sentiment
    logger.info("Assertion for empty social sentiment data passed.")

    # Test with social_sentiment_grok being None (should not print the section or error)
    none_sentiment_context = {key: val for key, val in mock_additional_context.items() if key != 'social_sentiment_grok'}
    prompt_none_sentiment = await builder.generate_prompt_with_data(
        mock_candles, "BTC/USDT", "marketAnalysis", additional_context=none_sentiment_context
    )
    assert "Social Sentiment/News (Grok):" not in prompt_none_sentiment
    logger.info("Assertion for None social sentiment data passed.")


    prompt_rm_no_sentiment = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "riskManagement",
        current_bias=0.6, current_confidence=0.75,
        additional_context={"entry_price": 2000, "current_profit_pct": 2.1} # No sentiment data here
    )
    logger.info(f"--- Risk Management Prompt (No Sentiment) ---\n{prompt_rm_no_sentiment}\n")
    assert "Social Sentiment/News (Grok):" not in prompt_rm_no_sentiment
    logger.info("Assertion for Risk Management prompt with no sentiment data passed.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_prompt_builder())

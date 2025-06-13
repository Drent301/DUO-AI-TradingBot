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
            final_prompt_parts.append("Analyze the following market data. Provide your trade intention (LONG, SHORT, HOLD), confidence level (0.0-1.0), and perceived bias (0.0-1.0).")
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
        primary_indicators = [
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'bb_lowerband', 'bb_middleband', 'bb_upperband',
            'ema_short', 'ema_long'
        ]
        for tf, df in candles_by_timeframe.items():
            if not df.empty:
                latest_row = df.iloc[-1]
                indicators_for_tf = {}
                for indicator_col in primary_indicators:
                    if indicator_col in df.columns and pd.notna(latest_row[indicator_col]):
                        indicators_for_tf[indicator_col] = latest_row[indicator_col]

                # Handle indicators with variable names like 'volume_mean_20'
                for col_name in df.columns:
                    if col_name.startswith('volume_mean') and pd.notna(latest_row[col_name]):
                        indicators_for_tf[col_name] = latest_row[col_name]

                if indicators_for_tf:
                    all_timeframe_indicators[tf] = indicators_for_tf

        if all_timeframe_indicators:
            final_prompt_parts.append("\nLatest Technical Indicators:")
            final_prompt_parts.append(json.dumps(all_timeframe_indicators, indent=2, default=str))


        final_prompt = "\n".join(final_prompt_parts)
        # logger.debug(f"Generated prompt for type '{prompt_type}' for {symbol}:\n{final_prompt}")
        return final_prompt


async def test_prompt_builder():
    builder = PromptBuilder()
    mock_candles = {
        '5m': object(), # In a real test, this would be a DataFrame
        '1h': object()  # In a real test, this would be a DataFrame
    }
    # Simulate DataFrames for to_string()
    # import pandas as pd # Already imported at the top
    mock_candles['5m'] = pd.DataFrame({
        'open': [10, 11, 12, 11.5, 12.5],
        'high': [10.5, 11.5, 12.5, 12, 13],
        'low': [9.5, 10.5, 11.5, 11, 12],
        'close': [11, 12, 11.5, 12.5, 12.8], # Last close 12.8
        'volume': [100, 120, 110, 130, 90],
        'rsi': [50, 55, 60, 58, 62.1],
        'macd': [0.1, 0.12, 0.15, 0.13, 0.16],
        'macdsignal': [0.09, 0.1, 0.11, 0.12, 0.13],
        'macdhist': [0.01, 0.02, 0.04, 0.01, 0.03],
        'bb_lowerband': [9, 10, 11, 10.5, 11.5],
        'bb_middleband': [10, 11, 12, 11.5, 12.5],
        'bb_upperband': [11, 12, 13, 12.5, 13.5],
        'ema_short': [10.8, 11.2, 11.8, 11.6, 12.2],
        'ema_long': [10.5, 10.8, 11.1, 11.3, 11.7],
        'volume_mean_10': [110, 115, 112, 118, 110.5] # Last value 110.5
    })
    # Add data for 1h timeframe as well
    mock_candles['1h'] = pd.DataFrame({
        'open': [1200, 1210, 1205],
        'high': [1250, 1220, 1215],
        'low': [1190, 1200, 1200],
        'close': [1210, 1205, 1212.7], # Last close 1212.7
        'volume': [1000, 1200, 1100],
        'rsi': [60.3, 55.2, 58.9],
        'ema_short': [1208, 1206, 1209.5],
        # This one intentionally has fewer indicators
    })


    prompt_ma = await builder.generate_prompt_with_data(mock_candles, "BTC/USDT", "marketAnalysis")
    logger.info(f"--- Market Analysis Prompt ---\n{prompt_ma}\n")

    # Assertions for the new technical indicators section
    assert "Latest Technical Indicators:" in prompt_ma, "Missing 'Latest Technical Indicators:' section title"
    assert '"5m": {' in prompt_ma, "Missing '5m' timeframe key in JSON"
    assert '"1h": {' in prompt_ma, "Missing '1h' timeframe key in JSON"
    assert '"rsi": 62.1' in prompt_ma, "Missing/incorrect rsi for 5m"
    assert '"macd": 0.16' in prompt_ma, "Missing/incorrect macd for 5m"
    assert '"volume_mean_10": 110.5' in prompt_ma, "Missing/incorrect volume_mean_10 for 5m"
    assert '"close": 12.8' not in prompt_ma, "OHLC data should not be in the JSON indicator block"
    assert '"rsi": 58.9' in prompt_ma, "Missing/incorrect rsi for 1h"
    assert '"ema_short": 1209.5' in prompt_ma, "Missing/incorrect ema_short for 1h"
    assert '"macd":' not in prompt_ma.split('"1h": {')[1], "MACD should not be in 1h data as it was not provided"


    logger.info("Assertions for Market Analysis prompt passed.")

    prompt_rm = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "riskManagement",
        current_bias=0.6, current_confidence=0.75,
        additional_context={"entry_price": 2000, "current_profit_pct": 2.1}
    )
    logger.info(f"--- Risk Management Prompt ---\n{prompt_rm}\n")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_prompt_builder())

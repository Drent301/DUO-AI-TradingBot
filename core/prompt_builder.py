import asyncio
import logging

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
    import pandas as pd
    mock_candles['5m'] = pd.DataFrame({
        'open': [10, 11, 12, 11.5, 12.5], 'high': [10.5, 11.5, 12.5, 12, 13],
        'low': [9.5, 10.5, 11.5, 11, 12], 'close': [11, 12, 11.5, 12.5, 12],
        'volume': [100, 120, 110, 130, 90]
    })
    mock_candles['1h'] = pd.DataFrame()


    prompt_ma = await builder.generate_prompt_with_data(mock_candles, "BTC/USDT", "marketAnalysis")
    logger.info(f"--- Market Analysis Prompt ---\n{prompt_ma}\n")

    prompt_rm = await builder.generate_prompt_with_data(
        mock_candles, "ETH/USDT", "riskManagement",
        current_bias=0.6, current_confidence=0.75,
        additional_context={"entry_price": 2000, "current_profit_pct": 2.1}
    )
    logger.info(f"--- Risk Management Prompt ---\n{prompt_rm}\n")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(test_prompt_builder())

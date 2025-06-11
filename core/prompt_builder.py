# core/prompt_builder.py
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Placeholder for PromptBuilder.
    This class is mocked in the tests for EntryDecider.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logger.info("Mock PromptBuilder initialized.")
        self.config = config if config else {}

    async def generate_prompt_with_data(
        self,
        candles_by_timeframe: Dict[str, pd.DataFrame],
        symbol: str,
        prompt_type: str, # e.g., 'marketAnalysis', 'exitSignal', 'reflection'
        current_bias: Optional[float] = None,
        current_confidence: Optional[float] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generates a prompt based on provided data and context.
        This is a mock implementation.
        """
        logger.info(f"Mock generate_prompt_with_data called for {symbol}, type: {prompt_type}")

        # Basic mock prompt, can be expanded if tests need more specific content
        mock_prompt = f"This is a mock prompt for {symbol} ({prompt_type})."
        if current_bias is not None:
            mock_prompt += f" Current bias: {current_bias:.2f}."
        if current_confidence is not None:
            mock_prompt += f" Current confidence: {current_confidence:.2f}."

        if additional_context:
            mock_prompt += f" Additional context: {str(additional_context)}."

        # Simulate async behavior if needed by caller, though for mocking it's often not critical
        # await asyncio.sleep(0.01)

        return mock_prompt

    def get_relevant_indicators(self, timeframe: str) -> List[str]:
        """
        Returns a list of relevant indicators for a given timeframe.
        Mock implementation.
        """
        # Example: Return a default set or vary by timeframe if necessary for tests
        default_indicators = [
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'bb_upperband', 'bb_middleband', 'bb_lowerband',
            'volume'
        ]
        logger.info(f"Mock get_relevant_indicators called for timeframe {timeframe}. Returning default list.")
        return default_indicators

    def _format_indicator_data(self, df: pd.DataFrame, indicators: List[str], max_candles: int) -> str:
        """
        Formats indicator data from a DataFrame into a string for the prompt.
        Mock implementation.
        """
        if df.empty:
            return "No indicator data available."

        subset = df[indicators].tail(max_candles)
        return subset.to_string(index=False)

    def _get_market_summary(self, candles_by_timeframe: Dict[str, pd.DataFrame]) -> str:
        """
        Generates a brief market summary.
        Mock implementation.
        """
        summary_parts = []
        for tf, df in candles_by_timeframe.items():
            if not df.empty:
                summary_parts.append(f"Timeframe {tf}: Last close {df['close'].iloc[-1]:.2f}, Volume {df['volume'].iloc[-1]:.0f}.")
        return " ".join(summary_parts) if summary_parts else "No market data available."

if __name__ == '__main__':
    # Example of using the mock PromptBuilder
    async def test_mock_prompt_builder():
        pb = PromptBuilder()

        # Create a mock dataframe
        mock_data = {'close': [10, 11, 12], 'volume': [100, 110, 120],
                     'rsi': [50,52,55], 'macd': [0.1,0.2,0.3], 'macdsignal': [0.1,0.15,0.18], 'macdhist': [0,0.05,0.12],
                     'bb_upperband': [13,14,15], 'bb_middleband': [11,12,13], 'bb_lowerband': [9,10,11]}
        mock_df = pd.DataFrame(mock_data)
        mock_candles = {'5m': mock_df}

        prompt = await pb.generate_prompt_with_data(
            candles_by_timeframe=mock_candles,
            symbol="BTC/USDT",
            prompt_type="marketAnalysis",
            current_bias=0.6,
            current_confidence=0.75,
            additional_context={"trend": "uptrend"}
        )
        print("Generated Mock Prompt:\n", prompt)

        indicators = pb.get_relevant_indicators("5m")
        print("\nRelevant Indicators (mock):", indicators)

        formatted_data = pb._format_indicator_data(mock_df, indicators, 3)
        print("\nFormatted Indicator Data (mock):\n", formatted_data)

        summary = pb._get_market_summary(mock_candles)
        print("\nMarket Summary (mock):\n", summary)

    # import asyncio # Already imported at top level for entry_decider
    asyncio.run(test_mock_prompt_builder())

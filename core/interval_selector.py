# core/interval_selector.py
import json
import logging
import os
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np # For random scores in test
import pandas as pd # For potential DataFrame type hinting in test
from datetime import datetime, timedelta # For potential DataFrame type hinting in test

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
MEMORY_PATH = os.path.join(MEMORY_DIR, 'interval_preference.json')
SUPPORTED_BASES = ['ETH', 'ZEN', 'BTC', 'USDT', 'EUR'] # Expanded based on your pair_whitelist

os.makedirs(MEMORY_DIR, exist_ok=True) # Ensure memory directory exists

# Helper function to read/write JSON, similar to your JS jsonHelpers
async def _read_json_async(filepath: str) -> Dict[str, Any]:
    def read_file_sync():
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    try:
        return await asyncio.to_thread(read_file_sync)
    except json.JSONDecodeError:
        logger.warning(f"JSON file {filepath} is corrupt or empty. Returning empty dict.")
        return {}
    except FileNotFoundError:
        return {} # File doesn't exist yet, return empty

async def _write_json_async(filepath: str, data: Dict[str, Any]):
    def write_file_sync():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    try:
        await asyncio.to_thread(write_file_sync)
    except Exception as e:
        logger.error(f"Error writing to JSON file {filepath}: {e}")


class IntervalSelector:
    """
    Detecteert en beheert de beste timeframe/interval voor handel op basis van leerbare scores.
    Vertaald van intervalSelector.js (promptBuilder_AANGEPAST.js).
    """
    def __init__(self):
        logger.info("IntervalSelector geïnitialiseerd.")

    async def detect_best_interval(self, base_token: str, strategy: str, token: str, available_intervals: List[str] = ['1m', '5m', '15m', '1h', '4h', '1d']) -> Optional[str]:
        """
        Detecteert de beste interval voor een token en strategie.
        Baseert zich op gesimuleerde backtest score, patroonscore en strategieprestaties.
        """
        if base_token not in SUPPORTED_BASES:
            logger.warning(f"Base token {base_token} is niet ondersteund. Kan beste interval niet detecteren.")
            return None

        interval_scores: Dict[str, Dict[str, float]] = {}

        # Dependencies for scores (mocked for now, in real scenario these come from other modules)
        # In a real Freqtrade integration, `getBacktestScore` would require running a backtest,
        # and `getStrategyPerformance` would come from `strategy_manager.py`.
        # `detectPatterns` would come from `cnn_patterns.py`.
        # from core.cnn_patterns import CNNPatterns # Re-import for type hint/mock
        # from core.strategy_manager import StrategyManager # Re-import for type hint/mock

        # Create mock instances for testing (replace with real instances in production)
        # mock_cnn_detector = CNNPatterns() # For detectPatterns
        # mock_strategy_manager = StrategyManager() # For getStrategyPerformance

        for interval in available_intervals:
            # Simulate fetching scores (replace with actual calls to backtest results, cnn_patterns, strategy_manager)
            backtest_score = self._get_mock_backtest_score(base_token, token, strategy, interval) # Simulate 0-1 range

            pattern_score = np.random.rand() # Simulate a random score for testing
            strat_score = np.random.rand() # Simulate a random score for testing


            # Combined score (weights from JS)
            combined_score = (0.5 * backtest_score) + (0.3 * pattern_score) + (0.2 * strat_score)
            interval_scores[interval] = {"backtestScore": backtest_score, "patternScore": pattern_score, "stratScore": strat_score, "combinedScore": combined_score}
            logger.debug(f"Interval {interval} score: {combined_score:.3f}")

        if not interval_scores:
            return None

        # Sort intervals by combinedScore (descending)
        sorted_intervals = sorted(interval_scores.items(), key=lambda item: item[1]["combinedScore"], reverse=True)
        best_interval = sorted_intervals[0][0]

        # Update memory
        await self.update_interval_memory(base_token, strategy, best_interval, interval_scores[best_interval]["combinedScore"])

        logger.info(f"Beste interval gedetecteerd voor {token} (strategie {strategy}): {best_interval}")
        return best_interval

    async def update_interval_memory(self, base_token: str, strategy: str, interval: str, score: float):
        """
        Werkt het geheugen voor intervalvoorkeuren bij.
        """
        memory = await _read_json_async(MEMORY_PATH)
        if base_token not in memory: memory[base_token] = {}
        if strategy not in memory[base_token]: memory[base_token][strategy] = {}

        memory[base_token][strategy][interval] = {"score": score, "last_updated": datetime.now().isoformat()}
        memory[base_token][strategy]["selected"] = interval # Mark as selected

        await _write_json_async(MEMORY_PATH, memory)
        logger.info(f"Interval geheugen bijgewerkt voor {base_token}/{strategy}: {interval} ({score:.3f}).")

    async def get_learned_interval(self, base_token: str, strategy: str) -> Optional[str]:
        """
        Haalt het laatst geleerde interval op voor een baseToken/strategie.
        """
        memory = await _read_json_async(MEMORY_PATH)
        return memory.get(base_token, {}).get(strategy, {}).get("selected")

    def is_consensus_between_intervals(self, scores_obj: Dict[str, Dict[str, float]], threshold: float = 0.1) -> bool:
        """
        Controleert of er consensus is tussen intervalscores (binnen een drempel).
        Vertaald van isConsensusBetweenIntervals in intervalSelector.js.
        """
        combined_scores = [item["combinedScore"] for item in scores_obj.values() if "combinedScore" in item]
        if len(combined_scores) < 2:
            return True # If only one or no scores, considered consensus

        sorted_scores = sorted(combined_scores, reverse=True)
        # Consensus if the difference between the top two scores is less than the threshold
        return (sorted_scores[0] - sorted_scores[1]) < threshold

    def _get_mock_backtest_score(self, base_token: str, token: str, strategy: str, interval: str) -> float:
        """ Simulate a backtest score (0-1). Replace with actual backtest results. """
        # In a real scenario, this would come from a Freqtrade backtest analysis.
        # For now, return a random score to simulate variability.
        # You can add logic here to make certain intervals perform better for specific tokens/strategies.
        seed = hash(f"{base_token}-{token}-{strategy}-{interval}") % 10**5 # Generate a "pseudo-random" score
        rng = np.random.default_rng(seed) # Use modern numpy random generator
        return rng.random() # Random float between 0.0 and 1.0


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import dotenv
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    async def run_test_interval_selector():
        selector = IntervalSelector()

        test_base_token = "ETH"
        test_strategy_id = "DUOAI_Strategy"
        test_token = "ETH/USDT"

        print("\n--- Test IntervalSelector ---")

        # Test detect_best_interval
        print(f"\nDetecting best interval for {test_token} (strategy {test_strategy_id})...")
        best_interval = await selector.detect_best_interval(test_base_token, test_strategy_id, test_token)
        print(f"Beste gedetecteerde interval: {best_interval}")
        assert best_interval is not None # Should find one

        # Test get_learned_interval
        learned_interval = await selector.get_learned_interval(test_base_token, test_strategy_id)
        print(f"Geleerd interval uit geheugen: {learned_interval}")
        assert learned_interval == best_interval # Should match

        # Test is_consensus_between_intervals
        mock_scores = {
            "1m": {"combinedScore": 0.85},
            "5m": {"combinedScore": 0.82},
            "15m": {"combinedScore": 0.50}
        }
        print(f"\nConsensus met 0.1 drempel: {selector.is_consensus_between_intervals(mock_scores, 0.1)}") # Should be True
        print(f"Consensus met 0.02 drempel: {selector.is_consensus_between_intervals(mock_scores, 0.02)}") # Should be False (0.85-0.82 = 0.03 > 0.02)

        # Clear memory for next test run if needed
        # await _write_json_async(MEMORY_PATH, {})

    asyncio.run(run_test_interval_selector())

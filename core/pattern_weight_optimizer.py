"""
PatternWeightOptimizer Module

Optimizes the weights of CNN patterns based on their historical performance.
It fetches performance metrics from PatternPerformanceAnalyzer and adjusts
pattern weights stored in ParamsManager to enhance strategy effectiveness.
"""
import logging
from typing import Dict, Any, Optional

from core.params_manager import ParamsManager # Actual import if running within a larger system
from core.pattern_performance_analyzer import PatternPerformanceAnalyzer # Actual import

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Allow external configuration

# Default configuration values
DEFAULT_MIN_PATTERN_WEIGHT = 0.1
DEFAULT_MAX_PATTERN_WEIGHT = 3.0 # Increased upper bound slightly
DEFAULT_PATTERN_WEIGHT_LEARNING_RATE = 0.05
DEFAULT_METRIC_TO_OPTIMIZE = 'win_rate_pct' # Can be 'win_rate_pct' or 'avg_profit_pct'
DEFAULT_LOW_PERF_THRESHOLD_WIN_RATE = 45.0 # %
DEFAULT_LOW_PERF_THRESHOLD_AVG_PROFIT = -0.5 # % (e.g. avg loss of 0.5%)
DEFAULT_HIGH_PERF_THRESHOLD_WIN_RATE = 60.0 # %
DEFAULT_HIGH_PERF_THRESHOLD_AVG_PROFIT = 1.0 # % (e.g. avg profit of 1%)
DEFAULT_MIN_TRADES_FOR_ADJUSTMENT = 10
DEFAULT_TARGET_STRATEGY_ID = "DUOAI_Strategy" # Default strategy to optimize weights for

class PatternWeightOptimizer:
    """
    Optimizes CNN pattern weights based on their performance.

    This class uses performance data (e.g., win rate, average profit) of patterns
    that contributed to trade entries to adjust their respective weights.
    The goal is to increase the influence of high-performing patterns and
    decrease that of underperforming ones, within configurable constraints.
    """
    def __init__(self, params_manager: Any, pattern_performance_analyzer: Any):
        """
        Initializes the PatternWeightOptimizer.

        Args:
            params_manager: An instance of ParamsManager to fetch configuration
                            and save updated pattern weights.
            pattern_performance_analyzer: An instance of PatternPerformanceAnalyzer
                                          to get pattern performance data.
        """
        self.params_manager = params_manager
        self.pattern_performance_analyzer = pattern_performance_analyzer
        # TODO: Make strategy_id_to_optimize configurable via params_manager for broader applicability
        self.strategy_id_to_optimize = DEFAULT_TARGET_STRATEGY_ID

        self._load_config()
        logger.info(f"PatternWeightOptimizer initialized for strategy '{self.strategy_id_to_optimize}'. Metric: '{self.metric_to_optimize}'. Learning rate: {self.learning_rate}.")

    def _load_config(self):
        """
        Loads optimizer-specific configuration from ParamsManager or uses defaults.
        These parameters control the optimization behavior.
        """
        # Fetching optimizer configurations as global parameters (strategy_id=None)
        self.min_pattern_weight = self.params_manager.get_param(
            "optimizerMinPatternWeight", strategy_id=None, default=DEFAULT_MIN_PATTERN_WEIGHT
        )
        self.max_pattern_weight = self.params_manager.get_param(
            "optimizerMaxPatternWeight", strategy_id=None, default=DEFAULT_MAX_PATTERN_WEIGHT
        )
        self.learning_rate = self.params_manager.get_param(
            "optimizerPatternWeightLearningRate", strategy_id=None, default=DEFAULT_PATTERN_WEIGHT_LEARNING_RATE
        )
        self.metric_to_optimize = self.params_manager.get_param(
            "optimizerPatternPerfMetric", strategy_id=None, default=DEFAULT_METRIC_TO_OPTIMIZE
        )
        self.low_perf_threshold_win_rate = self.params_manager.get_param(
            "optimizerLowPerfThresholdWinRate", strategy_id=None, default=DEFAULT_LOW_PERF_THRESHOLD_WIN_RATE
        )
        self.low_perf_threshold_avg_profit = self.params_manager.get_param(
            "optimizerLowPerfThresholdAvgProfit", strategy_id=None, default=DEFAULT_LOW_PERF_THRESHOLD_AVG_PROFIT
        )
        self.high_perf_threshold_win_rate = self.params_manager.get_param(
            "optimizerHighPerfThresholdWinRate", strategy_id=None, default=DEFAULT_HIGH_PERF_THRESHOLD_WIN_RATE
        )
        self.high_perf_threshold_avg_profit = self.params_manager.get_param(
            "optimizerHighPerfThresholdAvgProfit", strategy_id=None, default=DEFAULT_HIGH_PERF_THRESHOLD_AVG_PROFIT
        )
        self.min_trades_for_adjustment = self.params_manager.get_param(
            "optimizerMinTradesForWeightAdjustment", strategy_id=None, default=DEFAULT_MIN_TRADES_FOR_ADJUSTMENT
        )

        # Determine low and high performance thresholds based on the chosen metric
        if self.metric_to_optimize == 'win_rate_pct':
            self.low_perf_threshold = self.low_perf_threshold_win_rate
            self.high_perf_threshold = self.high_perf_threshold_win_rate
        elif self.metric_to_optimize == 'avg_profit_pct':
            self.low_perf_threshold = self.low_perf_threshold_avg_profit
            self.high_perf_threshold = self.high_perf_threshold_avg_profit
        else:
            logger.warning(f"Unknown metric_to_optimize: '{self.metric_to_optimize}'. Defaulting to win_rate_pct thresholds.")
            self.metric_to_optimize = 'win_rate_pct'
            self.low_perf_threshold = self.low_perf_threshold_win_rate
            self.high_perf_threshold = self.high_perf_threshold_win_rate


    async def optimize_pattern_weights(self) -> bool:
        """
        Optimizes the weights of CNN patterns based on their performance.

        Fetches performance metrics, iterates through CNN patterns, adjusts their
        weights based on configured thresholds and learning rates, and saves
        the updated weights back to ParamsManager.

        Returns:
            bool: True if any pattern weights were updated, False otherwise.
        """
        logger.info(f"Starting pattern weight optimization for strategy '{self.strategy_id_to_optimize}' using metric '{self.metric_to_optimize}'.")

        # Fetch current performance metrics for all patterns
        performance_metrics = self.pattern_performance_analyzer.analyze_pattern_performance()

        if not performance_metrics:
            logger.info("No performance metrics available. Skipping weight optimization.")
            return False

        updated_weights_count = 0
        fallback_cnn_pattern_weight = self.params_manager.get_param("cnnPatternWeight", strategy_id=self.strategy_id_to_optimize, default=1.0)

        for pattern_key, metrics in performance_metrics.items():
            # Only optimize CNN patterns (those starting with "cnn_")
            if not pattern_key.startswith("cnn_"):
                logger.debug(f"Skipping non-CNN pattern: {pattern_key}")
                continue

            # The pattern_key from analyzer (e.g., "cnn_5m_bullFlag") is used to form the weight parameter key.
            # Example: pattern_key "cnn_5m_bullFlag" -> weight_param_key "cnn_5m_bullFlag_weight".
            weight_param_key = f"{pattern_key}_weight"

            # Fetch the current weight for this specific pattern from ParamsManager.
            current_weight = self.params_manager.get_param(weight_param_key, strategy_id=self.strategy_id_to_optimize)

            if current_weight is None:
                # If a specific weight parameter (e.g., cnn_5m_bullFlag_weight) isn't found in params.json
                # for the strategy, it implies this pattern's weight hasn't been explicitly set and learned yet.
                # EntryDecider would use the general 'cnnPatternWeight' as a fallback in such scenarios.
                # Thus, for optimization, we start from this general fallback weight.
                current_weight = fallback_cnn_pattern_weight
                logger.info(f"Specific weight for '{weight_param_key}' not found for strategy '{self.strategy_id_to_optimize}'. Using general cnnPatternWeight: {current_weight:.3f} as starting point.")


            metric_value = metrics.get(self.metric_to_optimize)
            total_trades = metrics.get('total_trades', 0)

            if metric_value is None:
                logger.warning(f"Metric '{self.metric_to_optimize}' not found for pattern '{pattern_key}'. Skipping optimization for this pattern.")
                continue

            logger.info(f"Optimizing pattern: {pattern_key} (param: {weight_param_key}). Current weight: {current_weight:.3f}, Metric ({self.metric_to_optimize}): {metric_value:.2f}, Trades: {total_trades}")

            new_weight = current_weight

            # Adjust weight only if there's a significant number of trades for this pattern
            if total_trades >= self.min_trades_for_adjustment:
                # Increase weight for high-performing patterns
                if metric_value > self.high_perf_threshold:
                    new_weight = current_weight * (1 + self.learning_rate)
                    logger.info(f"  Performance HIGH for {pattern_key} ({self.metric_to_optimize}: {metric_value:.2f} > {self.high_perf_threshold:.2f}). Increasing weight.")
                # Decrease weight for low-performing patterns
                elif metric_value < self.low_perf_threshold:
                    new_weight = current_weight * (1 - self.learning_rate)
                    logger.info(f"  Performance LOW for {pattern_key} ({self.metric_to_optimize}: {metric_value:.2f} < {self.low_perf_threshold:.2f}). Decreasing weight.")
                # No change for moderately performing patterns
                else:
                    logger.info(f"  Performance MODERATE for {pattern_key} ({self.metric_to_optimize}: {metric_value:.2f}). Weight remains unchanged.")
            else:
                logger.info(f"  Not enough trades ({total_trades} < {self.min_trades_for_adjustment}) for {pattern_key}. Weight remains unchanged at {current_weight:.3f}.")

            # Apply min/max constraints to the new weight
            new_weight = max(self.min_pattern_weight, new_weight)
            new_weight = min(self.max_pattern_weight, new_weight)

            # Save the updated weight if it has changed meaningfully
            if abs(new_weight - current_weight) > 1e-5: # Using a small epsilon for float comparison
                await self.params_manager.set_param(weight_param_key, new_weight, strategy_id=self.strategy_id_to_optimize)
                logger.info(f"  SAVED: Updated weight for {pattern_key} (param: {weight_param_key}) from {current_weight:.3f} to {new_weight:.3f} for strategy '{self.strategy_id_to_optimize}'.")
                updated_weights_count += 1
            else:
                 logger.info(f"  No significant change in weight for {pattern_key} (param: {weight_param_key}) from {current_weight:.3f} (new calculated: {new_weight:.3f}). Not saving.")


        if updated_weights_count > 0:
            logger.info(f"Pattern weight optimization cycle finished. {updated_weights_count} pattern weights were updated for strategy '{self.strategy_id_to_optimize}'.")
            return True
        else:
            logger.info(f"Pattern weight optimization complete. No weights were updated for strategy '{self.strategy_id_to_optimize}'.")
            return False

if __name__ == '__main__':
    # This is a simplified example and requires proper ParamsManager and PatternPerformanceAnalyzer instances.
    # For a real test, you'd mock these or set up a test environment.

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Mock ParamsManager ---
    class MockParamsManager:
        def __init__(self):
            self.params = {
                "DUOAI_Strategy": {
                    "cnn_5m_bullFlag_weight": 1.0,
                    "cnn_1h_bearishEngulfing_weight": 0.8,
                    "cnnPatternWeight": 1.0 # Fallback
                },
                # Optimizer global config (strategy_id=None)
                "optimizerMinPatternWeight": 0.1,
                "optimizerMaxPatternWeight": 2.5,
                "optimizerPatternWeightLearningRate": 0.1, # Faster learning for test
                "optimizerPatternPerfMetric": 'win_rate_pct',
                "optimizerLowPerfThresholdWinRate": 40.0,
                "optimizerHighPerfThresholdWinRate": 65.0,
                "optimizerMinTradesForWeightAdjustment": 5
            }

        def get_param(self, key: str, strategy_id: Optional[str] = None, default: Any = None) -> Any:
            if strategy_id and strategy_id in self.params:
                return self.params[strategy_id].get(key, default)
            # For global params like optimizer settings, or if strategy_id is None
            return self.params.get(key, default)

        async def set_param(self, key: str, value: Any, strategy_id: Optional[str] = None):
            if strategy_id:
                if strategy_id not in self.params: self.params[strategy_id] = {}
                self.params[strategy_id][key] = value
                logger.info(f"MOCK_PM: Set '{key}' to {value} for strategy '{strategy_id}'")
            else:
                self.params[key] = value
                logger.info(f"MOCK_PM: Set global '{key}' to {value}")


    # --- Mock PatternPerformanceAnalyzer ---
    class MockPatternPerformanceAnalyzer:
        def analyze_pattern_performance(self) -> Dict[str, Dict[str, Any]]:
            # Return some dummy performance data
            return {
                "cnn_5m_bullFlag": { # High performance
                    'total_trades': 20, 'wins': 15, 'total_profit_pct': 0.30, # 30% total
                    'win_rate_pct': (15/20)*100, 'avg_profit_pct': (0.30/20)*100
                },
                "cnn_1h_bearishEngulfing": { # Low performance
                    'total_trades': 10, 'wins': 3, 'total_profit_pct': -0.05, # -5% total
                    'win_rate_pct': (3/10)*100, 'avg_profit_pct': (-0.05/10)*100
                },
                "cnn_15m_doji": { # Moderate performance, not enough trades
                    'total_trades': 3, 'wins': 2, 'total_profit_pct': 0.01,
                    'win_rate_pct': (2/3)*100, 'avg_profit_pct': (0.01/3)*100
                },
                "rule_morningStar": { # Should be ignored by optimizer
                    'total_trades': 50, 'wins': 30, 'total_profit_pct': 0.20,
                    'win_rate_pct': 60.0, 'avg_profit_pct': 0.4
                }
            }

    async def main_test():
        mock_pm = MockParamsManager()
        mock_ppa = MockPatternPerformanceAnalyzer()

        optimizer = PatternWeightOptimizer(params_manager=mock_pm, pattern_performance_analyzer=mock_ppa)
        await optimizer.optimize_pattern_weights()

        # Check updated weights in mock_pm
        logger.info("--- Final Weights in MockParamsManager ---")
        logger.info(json.dumps(mock_pm.params.get(DEFAULT_TARGET_STRATEGY_ID, {}), indent=2))

        # Expected:
        # cnn_5m_bullFlag: current=1.0, win_rate=75% (>65%), new_weight = 1.0 * (1+0.1) = 1.1
        # cnn_1h_bearishEngulfing: current=0.8, win_rate=30% (<40%), new_weight = 0.8 * (1-0.1) = 0.72
        # cnn_15m_doji: trades=3 (<5), weight unchanged (should start from fallback 1.0 if not set)
        #   - If cnn_15m_doji_weight is not in params, it uses fallback 1.0. Stays 1.0.
        #   - If it was, say, 0.5, it would stay 0.5.

        # Let's add cnn_15m_doji_weight to initial params to test its non-update path
        mock_pm.params[DEFAULT_TARGET_STRATEGY_ID]["cnn_15m_doji_weight"] = 0.9

        # Re-run with the new initial state for cnn_15m_doji_weight
        logger.info("\nRe-running optimization with cnn_15m_doji_weight initially set...")
        optimizer = PatternWeightOptimizer(params_manager=mock_pm, pattern_performance_analyzer=mock_ppa) # Re-init to reload config
        await optimizer.optimize_pattern_weights()
        logger.info("--- Final Weights after 2nd run ---")
        logger.info(json.dumps(mock_pm.params.get(DEFAULT_TARGET_STRATEGY_ID, {}), indent=2))

        assert abs(mock_pm.params[DEFAULT_TARGET_STRATEGY_ID]["cnn_5m_bullFlag_weight"] - 1.1 * (1+0.1)) < 1e-5 # Second increase
        assert abs(mock_pm.params[DEFAULT_TARGET_STRATEGY_ID]["cnn_1h_bearishEngulfing_weight"] - 0.72 * (1-0.1)) < 1e-5 # Second decrease
        assert abs(mock_pm.params[DEFAULT_TARGET_STRATEGY_ID]["cnn_15m_doji_weight"] - 0.9) < 1e-5 # Should remain 0.9


    if __name__ == "__main__":
        import asyncio
        asyncio.run(main_test())

```

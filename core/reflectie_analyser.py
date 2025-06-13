# core/reflectie_analyser.py
import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np # Voor gemiddelde
import asyncio # Voor async helper functies

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Padconfiguratie
MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
REFLECTIE_LOG = os.path.join(MEMORY_DIR, 'reflectie-logboek.json')
BIAS_OUTCOME_LOG = os.path.join(MEMORY_DIR, 'bias-outcome-log.json')
ANALYSE_LOG = os.path.join(MEMORY_DIR, 'reflectie-analyse.json')

os.makedirs(MEMORY_DIR, exist_ok=True)

from core.params_manager import ParamsManager

# Helperfuncties (kunnen static methods zijn of module-level blijven)
async def _load_json_async(filepath: str) -> List[Dict[str, Any]]:
    def read_file_sync():
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    try:
        content = await asyncio.to_thread(read_file_sync)
        if not isinstance(content, list):
            logger.warning(f"Content of {filepath} was not a list, returning empty list.")
            return []
        return content
    except json.JSONDecodeError:
        logger.warning(f"Kan {filepath} niet laden of bestand is corrupt, retourneer lege lijst.")
        return []
    except FileNotFoundError:
        logger.warning(f"Bestand {filepath} niet gevonden, retourneer lege lijst.")
        return []

async def _write_json_async(filepath: str, data: Any):
    def write_file_sync():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    try:
        await asyncio.to_thread(write_file_sync)
    except Exception as e:
        logger.error(f"Fout bij opslaan naar {filepath}: {e}")

def _validate_reflection(reflection: Dict[str, Any]) -> bool: # Renamed from validate_reflection
    if not reflection or not isinstance(reflection, dict):
        logger.debug('[ValidateReflection] Ongeldige reflectie: null of geen object')
        return False
    required_top_level = ['token', 'strategyId', 'combined_confidence', 'combined_bias']
    if not all(key in reflection for key in required_top_level):
        logger.debug(f'[ValidateReflection] Reflectie mist top-level vereiste velden: {reflection.keys()}')
        return False
    if not isinstance(reflection['combined_confidence'], (int, float)) or not (0 <= reflection['combined_confidence'] <= 1):
        logger.debug(f'[ValidateReflection] Ongeldige gecombineerde confidence: {reflection["combined_confidence"]}')
        return False
    if not isinstance(reflection['combined_bias'], (int, float)):
        logger.debug(f'[ValidateReflection] Ongeldige gecombineerde bias type: {reflection["combined_bias"]}')
        return False
    return True

def _mean(values: List[float]) -> float:
    if not values: return 0.0
    valid_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if not valid_values: return 0.0
    return float(np.mean(valid_values))

class ReflectieAnalyser:
    def __init__(self, params_manager: ParamsManager):
        self.params_manager = params_manager
        logger.info("ReflectieAnalyser geïnitialiseerd met ParamsManager.")

    async def analyse_reflecties(self, logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if logs is None:
            logs = await _load_json_async(REFLECTIE_LOG)
        if not isinstance(logs, list) or not logs:
            return {"biasScores": {}, "summary": {}}
        valid_logs = [log for log in logs if _validate_reflection(log)]
        if not valid_logs:
            return {"biasScores": {}, "summary": {}}

        bias_scores: Dict[str, Dict[str, Any]] = {}
        for log in valid_logs:
            strategy_id = log['strategyId']
            bias = log['combined_bias']
            confidence = log['combined_confidence']
            if strategy_id not in bias_scores:
                bias_scores[strategy_id] = {"totalBias": 0.0, "totalWeight": 0.0, "count": 0, "averageBias": 0.0, "confidence":0.0}
            weight = confidence
            bias_scores[strategy_id]["totalBias"] += bias * weight
            bias_scores[strategy_id]["totalWeight"] += weight
            bias_scores[strategy_id]["count"] += 1

        for strategy_id_key in bias_scores:
            entry = bias_scores[strategy_id_key]
            entry["averageBias"] = entry["totalBias"] / entry["totalWeight"] if entry["totalWeight"] > 0 else 0.0
            entry["confidence"] = entry["count"] / len(valid_logs) if len(valid_logs) > 0 else 0.0

        summary = {
            "totalReflections": len(valid_logs),
            "strategiesAnalyzed": len(bias_scores),
            "averageOverallBias": _mean([s['averageBias'] for s in bias_scores.values()]),
            "timestamp": datetime.now().isoformat()
        }
        analysis_data_to_store = {"biasScores": bias_scores, "summary": summary}
        await _write_json_async(ANALYSE_LOG, analysis_data_to_store)
        return analysis_data_to_store

    async def calculate_bias_score(self, reflections: Optional[List[Dict[str, Any]]] = None) -> float:
        if reflections is None: reflections = await _load_json_async(REFLECTIE_LOG)
        if not reflections: return 0.0
        valid_reflections = [r for r in reflections if _validate_reflection(r)]
        if not valid_reflections: return 0.0

        total_bias = sum(r['combined_bias'] * r['combined_confidence'] for r in valid_reflections)
        total_weight = sum(r['combined_confidence'] for r in valid_reflections)
        return total_bias / total_weight if total_weight > 0 else 0.0

    async def analyze_reflection_consistency(self, logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if logs is None: logs = await _load_json_async(REFLECTIE_LOG)
        if not logs or len(logs) < 2: return {"consistencyScore": 0.0, "details": {}}
        valid_logs = [log for log in logs if _validate_reflection(log)]
        if len(valid_logs) < 2: return {"consistencyScore": 0.0, "details": {}}

        details: Dict[str, Any] = {}
        total_variance_sum = 0.0
        strategy_count = 0
        grouped: Dict[str, List[float]] = {}
        for log in valid_logs:
            strategy_id = log['strategyId']
            grouped.setdefault(strategy_id, []).append(log['combined_bias'])

        for strategy_id_key, biases in grouped.items():
            if len(biases) < 2: continue
            variance = float(np.var(biases))
            details[strategy_id_key] = {"variance": variance, "sampleSize": len(biases)}
            total_variance_sum += variance
            strategy_count += 1

        average_variance = total_variance_sum / strategy_count if strategy_count > 0 else 0.0
        consistency_score = 1 / (1 + average_variance) if strategy_count > 0 else 0.0
        return {"consistencyScore": consistency_score, "details": details}

    async def _calculate_adjustment_score(self, strategy_id: str, bias: float, performance: Dict[str, Any]) -> float:
        win_rate = performance.get('winRate', 0.0)
        avg_profit = performance.get('avgProfit', 0.0)
        trade_count = performance.get('tradeCount', 0)

        avg_profit_scale_factor = self.params_manager.get_param("predictAdjust_avgProfitScaleFactor", strategy_id=strategy_id, default=600)
        avg_profit_score_min_max = self.params_manager.get_param("predictAdjust_avgProfitScoreMinMax", strategy_id=strategy_id, default=30)
        win_rate_weight = self.params_manager.get_param("predictAdjust_winRateWeight", strategy_id=strategy_id, default=50)
        trade_count_divisor = self.params_manager.get_param("predictAdjust_tradeCountDivisor", strategy_id=strategy_id, default=5)
        trade_count_score_max = self.params_manager.get_param("predictAdjust_tradeCountScoreMax", strategy_id=strategy_id, default=20)
        bias_high_threshold = self.params_manager.get_param("predictAdjust_biasHighThreshold", strategy_id=strategy_id, default=0.7)
        bias_low_threshold = self.params_manager.get_param("predictAdjust_biasLowThreshold", strategy_id=strategy_id, default=0.3)
        bias_influence_multiplier = self.params_manager.get_param("predictAdjust_biasInfluenceMultiplier", strategy_id=strategy_id, default=30)

        scaled_avg_profit_score = min(max(avg_profit * avg_profit_scale_factor, -avg_profit_score_min_max), avg_profit_score_min_max)
        performance_score = (win_rate * win_rate_weight) + scaled_avg_profit_score + (min(trade_count / trade_count_divisor if trade_count_divisor > 0 else 0, trade_count_score_max))

        adjustment_score = performance_score
        if bias > bias_high_threshold and performance_score > 50:
            adjustment_score += (bias - bias_high_threshold) * bias_influence_multiplier
        elif bias < bias_low_threshold and performance_score < 50:
            adjustment_score -= (bias_low_threshold - bias) * bias_influence_multiplier
        return max(0, min(100, adjustment_score))

    async def predict_strategy_adjustment(self, strategy: Dict[str, Any], bias: float, performance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not strategy or not isinstance(strategy, dict) or 'id' not in strategy: return None
        strategy_id = strategy['id']
        if not isinstance(bias, (int, float)) or np.isnan(bias): return None
        if not performance or not isinstance(performance, dict): return None

        parameters = strategy.get('parameters', {}) # These are the current parameters of the strategy being reflected upon
        adjustment_score = await self._calculate_adjustment_score(strategy_id, bias, performance)
        proposal = {"strategyId": strategy_id, "adjustments": {}, "confidence": 0.0, "currentAdjustmentScore": adjustment_score}

        adjustment_strengthen_threshold = self.params_manager.get_param("predictAdjust_adjStrengthenThreshold", strategy_id=strategy_id, default=70)
        adjustment_weaken_threshold = self.params_manager.get_param("predictAdjust_adjWeakenThreshold", strategy_id=strategy_id, default=30)
        param_strengthen_multiplier = self.params_manager.get_param("predictAdjust_paramStrengthenMultiplier", strategy_id=strategy_id, default=1.1)
        param_weaken_multiplier = self.params_manager.get_param("predictAdjust_paramWeakenMultiplier", strategy_id=strategy_id, default=0.9)
        param_ema_max = self.params_manager.get_param("predictAdjust_paramEmaMax", strategy_id=strategy_id, default=50)
        param_ema_min = self.params_manager.get_param("predictAdjust_paramEmaMin", strategy_id=strategy_id, default=5)
        # Assuming rsiThreshold is a single value that might be adjusted, or use separate buy/sell if strategy supports it
        param_rsi_buy_max = self.params_manager.get_param("predictAdjust_paramRsiBuyMax", strategy_id=strategy_id, default=40)
        param_rsi_buy_min = self.params_manager.get_param("predictAdjust_paramRsiBuyMin", strategy_id=strategy_id, default=15)
        param_rsi_sell_min = self.params_manager.get_param("predictAdjust_paramRsiSellMin", strategy_id=strategy_id, default=60)
        param_rsi_sell_max = self.params_manager.get_param("predictAdjust_paramRsiSellMax", strategy_id=strategy_id, default=85)


        if adjustment_score > adjustment_strengthen_threshold:
            proposal['adjustments'] = {"action": "strengthen", "parameterChanges": {}}
            if 'learned_ema_period' in parameters: # Check if the strategy is using this parameter
                proposal['adjustments']['parameterChanges']['emaPeriod'] = int(min(parameters['learned_ema_period'] * param_strengthen_multiplier, param_ema_max))
            if 'learned_rsi_threshold_buy' in parameters:
                 proposal['adjustments']['parameterChanges']['rsiThresholdBuy'] = int(max(parameters['learned_rsi_threshold_buy'] * param_weaken_multiplier, param_rsi_buy_min)) # RSI buy threshold decreases to strengthen
            if 'learned_rsi_threshold_sell' in parameters:
                 proposal['adjustments']['parameterChanges']['rsiThresholdSell'] = int(min(parameters['learned_rsi_threshold_sell'] * param_strengthen_multiplier, param_rsi_sell_max)) # RSI sell threshold increases
            proposal['confidence'] = adjustment_score / 100.0
        elif adjustment_score < adjustment_weaken_threshold:
            proposal['adjustments'] = {"action": "weaken", "parameterChanges": {}}
            if 'learned_ema_period' in parameters:
                proposal['adjustments']['parameterChanges']['emaPeriod'] = int(max(parameters['learned_ema_period'] * param_weaken_multiplier, param_ema_min))
            if 'learned_rsi_threshold_buy' in parameters:
                proposal['adjustments']['parameterChanges']['rsiThresholdBuy'] = int(min(parameters['learned_rsi_threshold_buy'] * param_strengthen_multiplier, param_rsi_buy_max)) # RSI buy threshold increases to weaken
            if 'learned_rsi_threshold_sell' in parameters:
                proposal['adjustments']['parameterChanges']['rsiThresholdSell'] = int(max(parameters['learned_rsi_threshold_sell'] * param_weaken_multiplier, param_rsi_sell_min)) # RSI sell threshold decreases
            proposal['confidence'] = (100.0 - adjustment_score) / 100.0
        else:
            proposal['adjustments'] = {"action": "maintain"}
            proposal['confidence'] = 0.5 + (abs(adjustment_score - 50.0)/100.0)

        logger.debug(f'[{strategy_id}] PredictStrategyAdjustment Proposal: {proposal}')
        return proposal

    async def analyze_timeframe_bias(self, reflections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if reflections is None: reflections = await _load_json_async(REFLECTIE_LOG)
        if not reflections: return {}
        valid_reflections = [r for r in reflections if _validate_reflection(r)]
        if not valid_reflections: return {}

        bias_by_timeframe: Dict[str, Dict[str, Any]] = {}
        for reflection in valid_reflections:
            strategy_id = reflection['strategyId']
            bias = reflection['combined_bias']
            confidence = reflection['combined_confidence']
            timeframe = reflection.get('timeframe') or reflection.get('trade_context', {}).get('timeframe')
            if not timeframe: continue

            bias_by_timeframe.setdefault(timeframe, {}).setdefault(strategy_id, {"totalBias": 0.0, "totalWeight": 0.0, "count": 0, "averageBias": 0.0, "confidence":0.0})

            weight = confidence
            bias_by_timeframe[timeframe][strategy_id]["totalBias"] += bias * weight
            bias_by_timeframe[timeframe][strategy_id]["totalWeight"] += weight
            bias_by_timeframe[timeframe][strategy_id]["count"] += 1

        for timeframe_key, strategies_on_tf in bias_by_timeframe.items():
            for strategy_id_key, entry in strategies_on_tf.items():
                entry["averageBias"] = entry["totalBias"] / entry["totalWeight"] if entry["totalWeight"] > 0 else 0.0
                total_strat_tf_reflections = sum(1 for r in valid_reflections if r.get('strategyId') == strategy_id_key and (r.get('timeframe') or r.get('trade_context', {}).get('timeframe')) == timeframe_key)
                entry["confidence"] = entry["count"] / total_strat_tf_reflections if total_strat_tf_reflections > 0 else 0.0
        return bias_by_timeframe

    async def generate_mutation_proposal(self, strategy_info: Dict[str, Any], current_bias: float, performance_data: Dict[str, Any], timeframe_bias_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if timeframe_bias_data is None:
            timeframe_bias_data = await self.analyze_timeframe_bias()

        # strategy_info should contain 'id' and 'parameters' (which holds current 'learned_ema_period', etc.)
        proposal = await self.predict_strategy_adjustment(strategy_info, current_bias, performance_data)
        if not proposal: return None

        proposal['timeframeSpecificAdjustments'] = {}
        strategy_id_str = strategy_info['id']

        if timeframe_bias_data:
            for timeframe, strategies_on_tf in timeframe_bias_data.items():
                if strategy_id_str in strategies_on_tf:
                    tf_bias_info = strategies_on_tf[strategy_id_str]
                    proposal['timeframeSpecificAdjustments'][timeframe] = {
                        "averageBias": tf_bias_info["averageBias"],
                        "confidence": tf_bias_info["confidence"],
                        "suggestedAction": "Monitor"
                    }

        proposal['rationale'] = {
            "overallBiasImpact": f"Overall AI bias: {current_bias:.2f}",
            "performanceImpact": f"Win Rate: {performance_data.get('winRate', 0):.2%}, Avg Profit: {performance_data.get('avgProfit', 0):.2%}, Trades: {performance_data.get('tradeCount', 0)}",
            "timeframeConsiderations": f"TF Biases: {json.dumps(proposal['timeframeSpecificAdjustments'])}" if proposal['timeframeSpecificAdjustments'] else 'No TF bias data.',
            "recommendedAction": proposal.get('adjustments', {}).get('action', 'maintain').capitalize(),
            "confidenceInProposal": proposal.get('confidence', 0.0)
        }

        # Log bias outcome
        bias_outcome_entry = {
            "strategyId": strategy_id_str, "timestamp": datetime.now().isoformat(),
            "overallBias": current_bias, "performance": performance_data,
            "proposedAction": proposal.get('adjustments', {}).get('action', 'maintain'),
            "proposedParameters": proposal.get('adjustments', {}).get('parameterChanges', {}),
            "proposalConfidence": proposal.get('confidence', 0.0)
        }
        try:
            existing_bias_outcomes = await _load_json_async(BIAS_OUTCOME_LOG)
            if not isinstance(existing_bias_outcomes, list): existing_bias_outcomes = []
            existing_bias_outcomes.append(bias_outcome_entry)
            await _write_json_async(BIAS_OUTCOME_LOG, existing_bias_outcomes)
        except Exception as e:
            logger.error(f"Fout bij bijwerken bias_outcome_log: {e}")

        logger.debug(f'[{strategy_id_str}] GenerateMutationProposal Result: {proposal}')
        return proposal

if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    mock_reflections_data = [
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.8, "combined_bias": 0.7, "timeframe": "5m", "trade_context": {"timeframe": "5m"}, "timestamp": "2025-06-11T10:00:00Z"},
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.6, "combined_bias": 0.4, "timeframe": "1h", "trade_context": {"timeframe": "1h"}, "timestamp": "2025-06-11T11:00:00Z"},
        {"token": "BTC/USDT", "strategyId": "Another_Strategy", "combined_confidence": 0.9, "combined_bias": 0.9, "timeframe": "15m", "trade_context": {"timeframe": "15m"}, "timestamp": "2025-06-11T12:00:00Z"},
    ]

    async def setup_mock_log():
        await _write_json_async(REFLECTIE_LOG, mock_reflections_data)
        await _write_json_async(ANALYSE_LOG, []) # Clear analyse log
        await _write_json_async(BIAS_OUTCOME_LOG, []) # Clear bias outcome log

    class MockParamsManager(ParamsManager): # Inherit for type hint compatibility if needed
        def __init__(self, custom_params=None):
            # super().__init__() # Skip actual file loading for mock
            self._params = custom_params if custom_params else {} # Store params directly
            self.default_params = self._get_default_params() # Keep access to defaults
            logger.info(f"MockParamsManager initialized with custom_params: {self.params}")

        def get_param(self, param_name: str, strategy_id: Optional[str] = None, default: Any = None) -> Any:
            # Simplified get: check strategy-specific, then global, then default from argument
            if strategy_id and "strategies" in self._params and strategy_id in self._params["strategies"] and \
               param_name in self._params["strategies"][strategy_id]:
                return self._params["strategies"][strategy_id][param_name]
            if "global" in self._params and param_name in self._params["global"]:
                return self._params["global"][param_name]

            # Fallback to provided default argument if key not found in custom mock params
            return default

        def _get_default_params(self) -> Dict[str, Any]: # Required by parent, not used by this mock's get_param
             return {"global": {}, "strategies": {}}


    async def run_test_reflectie_analyser():
        await setup_mock_log()

        mock_pm_default = MockParamsManager()
        analyser_default = ReflectieAnalyser(params_manager=mock_pm_default)

        analysis_result = await analyser_default.analyse_reflecties()
        duo_ai_bias = analysis_result.get("biasScores", {}).get("DUOAI_Strategy", {}).get("averageBias", 0.5)

        # Mock current strategy parameters as they would be fetched from ParamsManager
        mock_strategy_live_params = {
            "id": "DUOAI_Strategy",
            "parameters": {
                "learned_ema_period": 20,
                "learned_rsi_threshold_buy": 30,
                "learned_rsi_threshold_sell": 70
            }
        }
        mock_performance = {"winRate": 0.65, "avgProfit": 0.02, "tradeCount": 100}

        print(f"DUOAI_Strategy Bias for Default Test: {duo_ai_bias:.2f}")
        proposal = await analyser_default.generate_mutation_proposal(
            mock_strategy_live_params, duo_ai_bias, mock_performance)

        print("Mutation Proposal (Default Params):")
        print(json.dumps(proposal, indent=2))

        # Expected score: perf_score = (0.65*50) + min(max(0.02*600,-30),30) + min(100/5,20) = 32.5 + 12 + 20 = 64.5
        # Bias 0.59 (calculated from mock data). (0.59 - 0.5)*100 = 9. Not >0.7 or <0.3. So adj_score = 64.5
        # Default thresholds: Strengthen 70, Weaken 30. Action = maintain.
        assert proposal['adjustments']['action'] == 'maintain'
        assert abs(proposal['currentAdjustmentScore'] - 64.5) < 0.1 # Allow for float precision

    asyncio.run(run_test_reflectie_analyser())

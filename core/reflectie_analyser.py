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
BIAS_OUTCOME_LOG = os.path.join(MEMORY_DIR, 'bias-outcome-log.json') # Nieuw
ANALYSE_LOG = os.path.join(MEMORY_DIR, 'reflectie-analyse.json') # Nieuw

# Zorg dat de memory map bestaat
os.makedirs(MEMORY_DIR, exist_ok=True)

# Import ParamsManager for class usage
from core.params_manager import ParamsManager


# Helperfunctie voor JSON-laden (async) - module level or static method
async def _load_json_async(filepath: str) -> List[Dict[str, Any]]:
    def read_file_sync():
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    try:
        # Gebruik asyncio.to_thread voor blocking I/O in een async context
        content = await asyncio.to_thread(read_file_sync)
        if not isinstance(content, list): # Ensure it's a list, even if file contained a single dict or was malformed
            logger.warning(f"Content of {filepath} was not a list, returning empty list.")
            return []
        return content
    except json.JSONDecodeError:
        logger.warning(f"[ReflectieAnalyser] Kan {filepath} niet laden of bestand is corrupt, retourneer lege lijst.")
        return []
    except FileNotFoundError:
        logger.warning(f"[ReflectieAnalyser] Bestand {filepath} niet gevonden, retourneer lege lijst.")
        return []


# Helperfunctie voor JSON-opslaan (async)
async def _write_json_async(filepath: str, data: Any):
    def write_file_sync():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    try:
        await asyncio.to_thread(write_file_sync)
    except Exception as e:
        logger.error(f"[ReflectieAnalyser] Fout bij opslaan naar {filepath}: {e}")

# Helperfunctie voor validatie
def validate_reflection(reflection: Dict[str, Any]) -> bool:
    if not reflection or not isinstance(reflection, dict):
        logger.debug('[ValidateReflection] Ongeldige reflectie: null of geen object')
        return False
    # Vereiste velden: token, strategyId, gpt_response (met confidence), grok_response (met confidence)
    required_top_level = ['token', 'strategyId', 'combined_confidence', 'combined_bias']
    if not all(key in reflection for key in required_top_level):
        logger.debug(f'[ValidateReflection] Reflectie mist top-level vereiste velden: {reflection.keys()}')
        return False

    # Controleer de confidence en bias als nummers
    if not isinstance(reflection['combined_confidence'], (int, float)) or not (0 <= reflection['combined_confidence'] <= 1):
        logger.debug(f'[ValidateReflection] Ongeldige gecombineerde confidence: {reflection["combined_confidence"]}')
        return False
    # Bias can be 0-1 as per JS, or sometimes -1 to 1. Assuming 0-1 for now based on combined_bias calculation.
    if not isinstance(reflection['combined_bias'], (int, float)): # or not (0 <= reflection['combined_bias'] <= 1):
        logger.debug(f'[ValidateReflection] Ongeldige gecombineerde bias type: {reflection["combined_bias"]}')
        return False

    return True

# Helperfunctie voor gemiddelde
def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    valid_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
    if not valid_values:
        return 0.0
    return float(np.mean(valid_values))


async def analyse_reflecties(logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Geoptimaliseerde analyse van reflecties.
    Vertaald van analyseReflecties in reflectieAnalyser.js.
    Laadt logs van REFLECTIE_LOG als geen logs worden meegegeven.
    """
    if logs is None:
        logs = await _load_json_async(REFLECTIE_LOG)

    if not isinstance(logs, list) or not logs:
        logger.debug('[AnalyseReflecties] Geen logs beschikbaar of ongeldig type.')
        return {"biasScores": {}, "summary": {}}

    valid_logs = [log for log in logs if validate_reflection(log)]
    if not valid_logs:
        logger.debug('[AnalyseReflecties] Geen geldige reflecties na validatie.')
        return {"biasScores": {}, "summary": {}}

    bias_scores: Dict[str, Dict[str, Any]] = {}

    for log in valid_logs:
        strategy_id = log['strategyId']
        # Gebruik de gecombineerde bias en confidence die al in reflectie_lus zijn berekend
        bias = log['combined_bias']
        confidence = log['combined_confidence']

        if strategy_id not in bias_scores:
            bias_scores[strategy_id] = {"totalBias": 0.0, "totalWeight": 0.0, "count": 0, "averageBias": 0.0, "confidence":0.0}


        weight = confidence # Gewicht is de confidence score
        bias_scores[strategy_id]["totalBias"] += bias * weight
        bias_scores[strategy_id]["totalWeight"] += weight
        bias_scores[strategy_id]["count"] += 1

    for strategy_id_key in bias_scores: # Use strategy_id_key to avoid conflict
        entry = bias_scores[strategy_id_key]
        entry["averageBias"] = entry["totalBias"] / entry["totalWeight"] if entry["totalWeight"] > 0 else 0.0
        entry["confidence"] = entry["count"] / len(valid_logs) if len(valid_logs) > 0 else 0.0

    # Samenvatting
    total_reflections = len(valid_logs)
    strategies_analyzed = len(bias_scores)
    average_overall_bias = _mean([s['averageBias'] for s in bias_scores.values()])

    summary = {
        "totalReflections": total_reflections,
        "strategiesAnalyzed": strategies_analyzed,
        "averageOverallBias": average_overall_bias,
        "timestamp": datetime.now().isoformat()
    }

    analysis_data_to_store = {"biasScores": bias_scores, "summary": summary}
    await _write_json_async(ANALYSE_LOG, analysis_data_to_store) # Save analysis

    logger.debug(f'[AnalyseReflecties] Result: {analysis_data_to_store}')
    return analysis_data_to_store

async def calculate_bias_score(reflections: Optional[List[Dict[str, Any]]] = None) -> float:
    """
    Bereken gewogen bias-score voor een set reflecties.
    Vertaald van calculateBiasScore in reflectieAnalyser.js.
    """
    if reflections is None:
        reflections = await _load_json_async(REFLECTIE_LOG)

    if not reflections:
        logger.debug('[CalculateBiasScore] Geen reflecties beschikbaar.')
        return 0.0

    valid_reflections = [r for r in reflections if validate_reflection(r)]
    if not valid_reflections:
        logger.debug('[CalculateBiasScore] Geen geldige reflecties.')
        return 0.0

    total_bias = 0.0
    total_weight = 0.0

    for reflection in valid_reflections:
        bias = reflection['combined_bias']
        confidence = reflection['combined_confidence']
        weight = confidence # Gewicht is de confidence score
        total_bias += bias * weight
        total_weight += weight

    score = total_bias / total_weight if total_weight > 0 else 0.0
    logger.debug(f'[CalculateBiasScore] Result: {score}')
    return score

async def analyze_reflection_consistency(logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Analyseer consistentie van reflecties.
    Vertaald van analyzeReflectionConsistency in reflectieAnalyser.js.
    """
    if logs is None:
        logs = await _load_json_async(REFLECTIE_LOG)

    if not logs or len(logs) < 2:
        logger.debug('[AnalyzeReflectionConsistency] Te weinig logs voor consistentie-analyse.')
        return {"consistencyScore": 0.0, "details": {}}

    valid_logs = [log for log in logs if validate_reflection(log)]
    if len(valid_logs) < 2:
        logger.debug('[AnalyzeReflectionConsistency] Te weinig geldige reflecties.')
        return {"consistencyScore": 0.0, "details": {}}

    details: Dict[str, Any] = {}
    total_variance_sum = 0.0 # Renamed to avoid confusion
    strategy_count = 0

    grouped: Dict[str, List[float]] = {}
    for log in valid_logs:
        strategy_id = log['strategyId']
        if strategy_id not in grouped:
            grouped[strategy_id] = []
        grouped[strategy_id].append(log['combined_bias']) # Gebruik gecombineerde bias

    for strategy_id_key in grouped: # Use strategy_id_key
        biases = grouped[strategy_id_key]
        if len(biases) < 2: continue
        # avg_bias = _mean(biases) # Not used directly for consistency score
        variance = float(np.var(biases)) # NumPy's variance
        details[strategy_id_key] = {"variance": variance, "sampleSize": len(biases)}
        total_variance_sum += variance
        strategy_count += 1

    # Consistency score: higher is better. Inverse of average variance.
    average_variance = total_variance_sum / strategy_count if strategy_count > 0 else 0.0
    consistency_score = 1 / (1 + average_variance) if strategy_count > 0 else 0.0
    result = {"consistencyScore": consistency_score, "details": details}
    logger.debug(f'[AnalyzeReflectionConsistency] Result: {result}')
    return result

async def predict_strategy_adjustment(
    strategy: Dict[str, Any],
    bias: float,
    performance: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Geoptimaliseerde voorspelling van strategie-aanpassingen.
    Vertaald van predictStrategyAdjustment in reflectieAnalyser.js.
    """
    if not strategy or not isinstance(strategy, dict) or 'id' not in strategy:
        logger.debug('[PredictStrategyAdjustment] Ongeldige strategie.')
        return None
    if not isinstance(bias, (int, float)) or np.isnan(bias):
        logger.debug(f'[PredictStrategyAdjustment] Ongeldige bias: {bias}.')
        return None
    if not performance or not isinstance(performance, dict):
        logger.debug('[PredictStrategyAdjustment] Ongeldige performance.')
        return None

    strategy_id = strategy['id']
    parameters = strategy.get('parameters', {})
    win_rate = performance.get('winRate', 0.0)
    avg_profit = performance.get('avgProfit', 0.0) # Assuming this is a percentage like 0.02 for 2%
    trade_count = performance.get('tradeCount', 0)

    # Score strategie op basis van bias en performance
    # Performance score: 0-100 range. Win rate (50%), Avg Profit (30% -> scale it, e.g. 1% profit = 10 points), Trade count (20%)
    # Let's make avg_profit impact more direct: e.g. cap at 5% profit = 30 points. 1% = 6 points.
    scaled_avg_profit_score = min(max(avg_profit * 600, -30), 30) # e.g. 1% profit = 6 points, max 30 points for 5%

    performance_score = (win_rate * 50) + scaled_avg_profit_score + (min(trade_count / 5, 20)) # Max 20 points for trade_count (e.g. 100 trades = 20 points)

    # Combine bias (0-1, needs scaling if it's e.g. -1 to 1) with performance score
    # Assuming bias is 0-1, where 0.5 is neutral. We can map it to -50 to 50.
    # (bias - 0.5) * 100 gives -50 to 50.
    # For adjustment_score, let's use a simple weighted average.
    # Bias influence can be direct on adjustment direction.

    adjustment_score = performance_score # Start with performance

    # Modify score based on bias. If bias is strongly positive (>0.7) and perf is good, boost.
    # If bias is strongly negative (<0.3) and perf is bad, penalize further.
    if bias > 0.7 and performance_score > 50:
        adjustment_score += (bias - 0.7) * 30 # Max boost of (1-0.7)*30 = 9
    elif bias < 0.3 and performance_score < 50:
        adjustment_score -= (0.3 - bias) * 30 # Max penalty of (0.3-0)*30 = 9

    adjustment_score = max(0, min(100, adjustment_score)) # Clamp to 0-100


    proposal = {"strategyId": strategy_id, "adjustments": {}, "confidence": 0.0, "currentAdjustmentScore": adjustment_score}


    # Voorstellen op basis van score
    if adjustment_score > 70: # Hogere drempel voor positieve aanpassing
        proposal['adjustments'] = {
            "action": "strengthen", # More descriptive
            "parameterChanges": {}
        }
        if 'emaPeriod' in parameters and isinstance(parameters['emaPeriod'], (int, float)):
            proposal['adjustments']['parameterChanges']['emaPeriod'] = int(min(parameters['emaPeriod'] * 1.1, 50)) # Proportional change
        if 'rsiThreshold' in parameters and isinstance(parameters['rsiThreshold'], (int, float)):
            proposal['adjustments']['parameterChanges']['rsiThreshold'] = int(min(parameters['rsiThreshold'] * 1.1, 85))
        proposal['confidence'] = adjustment_score / 100
    elif adjustment_score < 30: # Lagere drempel voor negatieve aanpassing
        proposal['adjustments'] = {
            "action": "weaken", # More descriptive
            "parameterChanges": {}
        }
        if 'emaPeriod' in parameters and isinstance(parameters['emaPeriod'], (int, float)):
            proposal['adjustments']['parameterChanges']['emaPeriod'] = int(max(parameters['emaPeriod'] * 0.9, 5))
        if 'rsiThreshold' in parameters and isinstance(parameters['rsiThreshold'], (int, float)):
            proposal['adjustments']['parameterChanges']['rsiThreshold'] = int(max(parameters['rsiThreshold'] * 0.9, 15))
        proposal['confidence'] = (100 - adjustment_score) / 100 # Hoe lager score, hoe hoger confidence in negatieve aanpassing
    else:
        proposal['adjustments'] = {"action": "maintain"}
        proposal['confidence'] = 0.5 + (abs(adjustment_score - 50)/100) # Confidence in maintaining is higher if score is near 50

    logger.debug(f'[PredictStrategyAdjustment] Proposal: {proposal}')
    return proposal

async def analyze_timeframe_bias(reflections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Analyseer timeframe-specifieke reflecties.
    Vertaald van analyzeTimeframeBias in reflectieAnalyser.js.
    """
    if reflections is None:
        reflections = await _load_json_async(REFLECTIE_LOG)

    if not reflections:
        logger.debug('[AnalyzeTimeframeBias] Geen reflecties beschikbaar.')
        return {}

    valid_reflections = [r for r in reflections if validate_reflection(r)]
    if not valid_reflections:
        logger.debug('[AnalyzeTimeframeBias] Geen geldige reflecties.')
        return {}

    bias_by_timeframe: Dict[str, Dict[str, Any]] = {}
    for reflection in valid_reflections:
        strategy_id = reflection['strategyId']
        bias = reflection['combined_bias']
        confidence = reflection['combined_confidence']
        # Aanname: timeframe is in trade_context, of direct in reflection als het een algemene analyse is
        timeframe = reflection.get('timeframe') or reflection.get('trade_context', {}).get('timeframe')


        if not timeframe: continue

        if timeframe not in bias_by_timeframe:
            bias_by_timeframe[timeframe] = {}
        if strategy_id not in bias_by_timeframe[timeframe]:
            bias_by_timeframe[timeframe][strategy_id] = {"totalBias": 0.0, "totalWeight": 0.0, "count": 0, "averageBias": 0.0, "confidence":0.0}


        weight = confidence
        bias_by_timeframe[timeframe][strategy_id]["totalBias"] += bias * weight
        bias_by_timeframe[timeframe][strategy_id]["totalWeight"] += weight
        bias_by_timeframe[timeframe][strategy_id]["count"] += 1

    for timeframe_key in bias_by_timeframe: # Use timeframe_key
        for strategy_id_key in bias_by_timeframe[timeframe_key]: # Use strategy_id_key
            entry = bias_by_timeframe[timeframe_key][strategy_id_key]
            entry["averageBias"] = entry["totalBias"] / entry["totalWeight"] if entry["totalWeight"] > 0 else 0.0
            # Confidence in this specific timeframe's bias for the strategy
            entry["confidence"] = entry["count"] / sum(1 for r in valid_reflections if r.get('strategyId') == strategy_id_key and (r.get('timeframe') or r.get('trade_context', {}).get('timeframe')) == timeframe_key) if sum(1 for r in valid_reflections if r.get('strategyId') == strategy_id_key and (r.get('timeframe') or r.get('trade_context', {}).get('timeframe')) == timeframe_key) > 0 else 0.0


    logger.debug(f'[AnalyzeTimeframeBias] Result: {bias_by_timeframe}')
    return bias_by_timeframe


class ReflectieAnalyser:
    def __init__(self, params_manager: ParamsManager):
        self.params_manager = params_manager
        logger.info("ReflectieAnalyser geïnitialiseerd met ParamsManager.")

    async def analyse_reflecties(self, logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Geoptimaliseerde analyse van reflecties.
        Laadt logs van REFLECTIE_LOG als geen logs worden meegegeven.
        """
        if logs is None:
            logs = await _load_json_async(REFLECTIE_LOG)

        if not isinstance(logs, list) or not logs:
            logger.debug('[AnalyseReflecties] Geen logs beschikbaar of ongeldig type.')
            return {"biasScores": {}, "summary": {}}

        valid_logs = [log for log in logs if validate_reflection(log)]
        if not valid_logs:
            logger.debug('[AnalyseReflecties] Geen geldige reflecties na validatie.')
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

        total_reflections = len(valid_logs)
        strategies_analyzed = len(bias_scores)
        average_overall_bias = _mean([s['averageBias'] for s in bias_scores.values()])

        summary = {
            "totalReflections": total_reflections,
            "strategiesAnalyzed": strategies_analyzed,
            "averageOverallBias": average_overall_bias,
            "timestamp": datetime.now().isoformat()
        }
        analysis_data_to_store = {"biasScores": bias_scores, "summary": summary}
        await _write_json_async(ANALYSE_LOG, analysis_data_to_store)
        logger.debug(f'[AnalyseReflecties] Result: {analysis_data_to_store}')
        return analysis_data_to_store

    async def calculate_bias_score(self, reflections: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Bereken gewogen bias-score voor een set reflecties.
        """
        if reflections is None:
            reflections = await _load_json_async(REFLECTIE_LOG)
        if not reflections:
            logger.debug('[CalculateBiasScore] Geen reflecties beschikbaar.')
            return 0.0
        valid_reflections = [r for r in reflections if validate_reflection(r)]
        if not valid_reflections:
            logger.debug('[CalculateBiasScore] Geen geldige reflecties.')
            return 0.0
        total_bias = 0.0
        total_weight = 0.0
        for reflection in valid_reflections:
            bias = reflection['combined_bias']
            confidence = reflection['combined_confidence']
            weight = confidence
            total_bias += bias * weight
            total_weight += weight
        score = total_bias / total_weight if total_weight > 0 else 0.0
        logger.debug(f'[CalculateBiasScore] Result: {score}')
        return score

    async def analyze_reflection_consistency(self, logs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyseer consistentie van reflecties.
        """
        if logs is None:
            logs = await _load_json_async(REFLECTIE_LOG)
        if not logs or len(logs) < 2:
            logger.debug('[AnalyzeReflectionConsistency] Te weinig logs voor consistentie-analyse.')
            return {"consistencyScore": 0.0, "details": {}}
        valid_logs = [log for log in logs if validate_reflection(log)]
        if len(valid_logs) < 2:
            logger.debug('[AnalyzeReflectionConsistency] Te weinig geldige reflecties.')
            return {"consistencyScore": 0.0, "details": {}}
        details: Dict[str, Any] = {}
        total_variance_sum = 0.0
        strategy_count = 0
        grouped: Dict[str, List[float]] = {}
        for log in valid_logs:
            strategy_id = log['strategyId']
            if strategy_id not in grouped:
                grouped[strategy_id] = []
            grouped[strategy_id].append(log['combined_bias'])
        for strategy_id_key in grouped:
            biases = grouped[strategy_id_key]
            if len(biases) < 2: continue
            variance = float(np.var(biases))
            details[strategy_id_key] = {"variance": variance, "sampleSize": len(biases)}
            total_variance_sum += variance
            strategy_count += 1
        average_variance = total_variance_sum / strategy_count if strategy_count > 0 else 0.0
        consistency_score = 1 / (1 + average_variance) if strategy_count > 0 else 0.0
        result = {"consistencyScore": consistency_score, "details": details}
        logger.debug(f'[AnalyzeReflectionConsistency] Result: {result}')
        return result

    async def _calculate_adjustment_score(
        self,
        strategy_id: str, # Added strategy_id for fetching params
        bias: float,
        performance: Dict[str, Any]
    ) -> float:
        win_rate = performance.get('winRate', 0.0)
        avg_profit = performance.get('avgProfit', 0.0)
        trade_count = performance.get('tradeCount', 0)

        # Fetch constants from ParamsManager
        avg_profit_scale_factor = self.params_manager.get_param("predictAdjust_avgProfitScaleFactor", strategy_id=strategy_id, default=600)
        avg_profit_score_min_max = self.params_manager.get_param("predictAdjust_avgProfitScoreMinMax", strategy_id=strategy_id, default=30)
        win_rate_weight = self.params_manager.get_param("predictAdjust_winRateWeight", strategy_id=strategy_id, default=50)
        trade_count_divisor = self.params_manager.get_param("predictAdjust_tradeCountDivisor", strategy_id=strategy_id, default=5)
        trade_count_score_max = self.params_manager.get_param("predictAdjust_tradeCountScoreMax", strategy_id=strategy_id, default=20)
        bias_high_threshold = self.params_manager.get_param("predictAdjust_biasHighThreshold", strategy_id=strategy_id, default=0.7)
        bias_low_threshold = self.params_manager.get_param("predictAdjust_biasLowThreshold", strategy_id=strategy_id, default=0.3)
        bias_influence_multiplier = self.params_manager.get_param("predictAdjust_biasInfluenceMultiplier", strategy_id=strategy_id, default=30)

        logger.info(f"[{strategy_id}] Using adjustment score params: avgProfitScaleFactor={avg_profit_scale_factor}, avgProfitScoreMinMax={avg_profit_score_min_max}, "
                    f"winRateWeight={win_rate_weight}, tradeCountDivisor={trade_count_divisor}, tradeCountScoreMax={trade_count_score_max}, "
                    f"biasHighThreshold={bias_high_threshold}, biasLowThreshold={bias_low_threshold}, biasInfluenceMultiplier={bias_influence_multiplier}")

        scaled_avg_profit_score = min(max(avg_profit * avg_profit_scale_factor, -avg_profit_score_min_max), avg_profit_score_min_max)
        performance_score = (win_rate * win_rate_weight) + scaled_avg_profit_score + (min(trade_count / trade_count_divisor if trade_count_divisor > 0 else 0, trade_count_score_max))

        adjustment_score = performance_score
        if bias > bias_high_threshold and performance_score > 50: # Assuming 50 is a neutral performance score
            adjustment_score += (bias - bias_high_threshold) * bias_influence_multiplier
        elif bias < bias_low_threshold and performance_score < 50:
            adjustment_score -= (bias_low_threshold - bias) * bias_influence_multiplier

        return max(0, min(100, adjustment_score))

    async def predict_strategy_adjustment(
        self,
        strategy: Dict[str, Any],
        bias: float,
        performance: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not strategy or not isinstance(strategy, dict) or 'id' not in strategy:
            logger.debug('[PredictStrategyAdjustment] Ongeldige strategie.')
            return None
        strategy_id = strategy['id']
        if not isinstance(bias, (int, float)) or np.isnan(bias):
            logger.debug(f'[{strategy_id}] Ongeldige bias: {bias}.')
            return None
        if not performance or not isinstance(performance, dict):
            logger.debug(f'[{strategy_id}] Ongeldige performance.')
            return None

        parameters = strategy.get('parameters', {})
        adjustment_score = await self._calculate_adjustment_score(strategy_id, bias, performance)

        proposal = {"strategyId": strategy_id, "adjustments": {}, "confidence": 0.0, "currentAdjustmentScore": adjustment_score}

        # Fetch thresholds and multipliers for parameter adjustment
        adjustment_strengthen_threshold = self.params_manager.get_param("predictAdjust_adjStrengthenThreshold", strategy_id=strategy_id, default=70)
        adjustment_weaken_threshold = self.params_manager.get_param("predictAdjust_adjWeakenThreshold", strategy_id=strategy_id, default=30)
        param_strengthen_multiplier = self.params_manager.get_param("predictAdjust_paramStrengthenMultiplier", strategy_id=strategy_id, default=1.1)
        param_weaken_multiplier = self.params_manager.get_param("predictAdjust_paramWeakenMultiplier", strategy_id=strategy_id, default=0.9)
        param_ema_max = self.params_manager.get_param("predictAdjust_paramEmaMax", strategy_id=strategy_id, default=50)
        param_ema_min = self.params_manager.get_param("predictAdjust_paramEmaMin", strategy_id=strategy_id, default=5)
        param_rsi_max = self.params_manager.get_param("predictAdjust_paramRsiMax", strategy_id=strategy_id, default=85)
        param_rsi_min = self.params_manager.get_param("predictAdjust_paramRsiMin", strategy_id=strategy_id, default=15)

        logger.info(f"[{strategy_id}] Using parameter adjustment thresholds/multipliers: adjStrengthenThreshold={adjustment_strengthen_threshold}, adjWeakenThreshold={adjustment_weaken_threshold}, "
                    f"paramStrengthenMultiplier={param_strengthen_multiplier}, paramWeakenMultiplier={param_weaken_multiplier}, emaMax={param_ema_max}, emaMin={param_ema_min}, "
                    f"rsiMax={param_rsi_max}, rsiMin={param_rsi_min}")

        if adjustment_score > adjustment_strengthen_threshold:
            proposal['adjustments'] = {"action": "strengthen", "parameterChanges": {}}
            if 'emaPeriod' in parameters and isinstance(parameters['emaPeriod'], (int, float)):
                proposal['adjustments']['parameterChanges']['emaPeriod'] = int(min(parameters['emaPeriod'] * param_strengthen_multiplier, param_ema_max))
            if 'rsiThreshold' in parameters and isinstance(parameters['rsiThreshold'], (int, float)):
                proposal['adjustments']['parameterChanges']['rsiThreshold'] = int(min(parameters['rsiThreshold'] * param_strengthen_multiplier, param_rsi_max))
            proposal['confidence'] = adjustment_score / 100
        elif adjustment_score < adjustment_weaken_threshold:
            proposal['adjustments'] = {"action": "weaken", "parameterChanges": {}}
            if 'emaPeriod' in parameters and isinstance(parameters['emaPeriod'], (int, float)):
                proposal['adjustments']['parameterChanges']['emaPeriod'] = int(max(parameters['emaPeriod'] * param_weaken_multiplier, param_ema_min))
            if 'rsiThreshold' in parameters and isinstance(parameters['rsiThreshold'], (int, float)):
                proposal['adjustments']['parameterChanges']['rsiThreshold'] = int(max(parameters['rsiThreshold'] * param_weaken_multiplier, param_rsi_min))
            proposal['confidence'] = (100 - adjustment_score) / 100
        else:
            proposal['adjustments'] = {"action": "maintain"}
            proposal['confidence'] = 0.5 + (abs(adjustment_score - 50)/100)

        logger.debug(f'[{strategy_id}] PredictStrategyAdjustment Proposal: {proposal}')
        return proposal

    async def analyze_timeframe_bias(self, reflections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyseer timeframe-specifieke reflecties.
        """
        if reflections is None:
            reflections = await _load_json_async(REFLECTIE_LOG)
        if not reflections:
            logger.debug('[AnalyzeTimeframeBias] Geen reflecties beschikbaar.')
            return {}
        valid_reflections = [r for r in reflections if validate_reflection(r)]
        if not valid_reflections:
            logger.debug('[AnalyzeTimeframeBias] Geen geldige reflecties.')
            return {}
        bias_by_timeframe: Dict[str, Dict[str, Any]] = {}
        for reflection in valid_reflections:
            strategy_id = reflection['strategyId']
            bias = reflection['combined_bias']
            confidence = reflection['combined_confidence']
            timeframe = reflection.get('timeframe') or reflection.get('trade_context', {}).get('timeframe')
            if not timeframe: continue
            if timeframe not in bias_by_timeframe:
                bias_by_timeframe[timeframe] = {}
            if strategy_id not in bias_by_timeframe[timeframe]:
                bias_by_timeframe[timeframe][strategy_id] = {"totalBias": 0.0, "totalWeight": 0.0, "count": 0, "averageBias": 0.0, "confidence":0.0}
            weight = confidence
            bias_by_timeframe[timeframe][strategy_id]["totalBias"] += bias * weight
            bias_by_timeframe[timeframe][strategy_id]["totalWeight"] += weight
            bias_by_timeframe[timeframe][strategy_id]["count"] += 1
        for timeframe_key in bias_by_timeframe:
            for strategy_id_key in bias_by_timeframe[timeframe_key]:
                entry = bias_by_timeframe[timeframe_key][strategy_id_key]
                entry["averageBias"] = entry["totalBias"] / entry["totalWeight"] if entry["totalWeight"] > 0 else 0.0
                entry["confidence"] = entry["count"] / sum(1 for r in valid_reflections if r.get('strategyId') == strategy_id_key and (r.get('timeframe') or r.get('trade_context', {}).get('timeframe')) == timeframe_key) if sum(1 for r in valid_reflections if r.get('strategyId') == strategy_id_key and (r.get('timeframe') or r.get('trade_context', {}).get('timeframe')) == timeframe_key) > 0 else 0.0
        logger.debug(f'[AnalyzeTimeframeBias] Result: {bias_by_timeframe}')
        return bias_by_timeframe

    async def generate_mutation_proposal(
        self, # Added self
        strategy: Dict[str, Any],
        bias: float, # Overall bias for the strategy
        performance: Dict[str, Any],
        timeframe_bias_analysis: Optional[Dict[str, Any]] = None # Result from analyze_timeframe_bias
    ) -> Optional[Dict[str, Any]]:
        """
        Genereer mutatievoorstel met timeframe-data.
        """
        if timeframe_bias_analysis is None:
            timeframe_bias_analysis = await self.analyze_timeframe_bias() # Call as method

        proposal = await self.predict_strategy_adjustment(strategy, bias, performance) # Call as method
        if not proposal: return None

        proposal['timeframeSpecificAdjustments'] = {}
        strategy_id_str = strategy['id']

        if timeframe_bias_analysis:
            for timeframe, strategies_on_tf in timeframe_bias_analysis.items():
                if strategy_id_str in strategies_on_tf:
                    tf_bias_info = strategies_on_tf[strategy_id_str]
                    proposal['timeframeSpecificAdjustments'][timeframe] = {
                        "averageBias": tf_bias_info["averageBias"],
                        "confidence": tf_bias_info["confidence"],
                        "suggestedAction": "Monitor"
                    }

        proposal['rationale'] = {
            "overallBiasImpact": f"Overall AI bias towards strategy: {bias:.2f}",
            "performanceImpact": f"Win Rate: {performance.get('winRate', 0):.2%}, Avg Profit: {performance.get('avgProfit', 0):.2%}, Trades: {performance.get('tradeCount', 0)}",
            "timeframeConsiderations": f"Timeframe-specific biases: {json.dumps(proposal['timeframeSpecificAdjustments'])}" if proposal['timeframeSpecificAdjustments'] else 'No specific timeframe bias data applied to this proposal.',
            "recommendedAction": proposal.get('adjustments', {}).get('action', 'maintain').capitalize(),
            "confidenceInProposal": proposal.get('confidence', 0.0)
        }

        bias_outcome_entry = {
            "strategyId": strategy_id_str,
            "timestamp": datetime.now().isoformat(),
            "overallBias": bias,
            "performance": performance,
            "proposedAction": proposal.get('adjustments', {}).get('action', 'maintain'),
            "proposedParameters": proposal.get('adjustments', {}).get('parameterChanges', {}),
            "proposalConfidence": proposal.get('confidence', 0.0)
        }
        existing_bias_outcomes = await _load_json_async(BIAS_OUTCOME_LOG)
        if not isinstance(existing_bias_outcomes, list): existing_bias_outcomes = []
        existing_bias_outcomes.append(bias_outcome_entry)
        await _write_json_async(BIAS_OUTCOME_LOG, existing_bias_outcomes)
        logger.debug(f'[{strategy_id_str}] GenerateMutationProposal Result: {proposal}')
        return proposal

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    mock_reflections_data = [
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.8, "combined_bias": 0.7, "timeframe": "5m", "trade_context": {"timeframe": "5m"}, "timestamp": "2025-06-11T10:00:00Z"},
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.6, "combined_bias": 0.4, "timeframe": "1h", "trade_context": {"timeframe": "1h"}, "timestamp": "2025-06-11T11:00:00Z"},
        {"token": "BTC/USDT", "strategyId": "Another_Strategy", "combined_confidence": 0.9, "combined_bias": 0.9, "timeframe": "15m", "trade_context": {"timeframe": "15m"}, "timestamp": "2025-06-11T12:00:00Z"},
    ]

    async def setup_mock_log():
        await _write_json_async(REFLECTIE_LOG, mock_reflections_data)
        await _write_json_async(ANALYSE_LOG, [])
        await _write_json_async(BIAS_OUTCOME_LOG, [])

    class MockParamsManager:
        def __init__(self, custom_params=None):
            self.params = custom_params if custom_params else {}
            logger.info(f"MockParamsManager initialized with params: {self.params}")

        def get_param(self, param_name: str, strategy_id: Optional[str] = None, default: Any = None) -> Any:
            # Simple mock: strategy_id not used here, but could be for more complex mocks
            value = self.params.get(param_name, default)
            # logger.info(f"[MockParamsManager] get_param: '{param_name}' (Strategy: {strategy_id}) -> Value: {value} (Default was: {default})")
            return value

    async def run_test_reflectie_analyser():
        await setup_mock_log()

        # --- Test Scenario A: Default Values ---
        print("\n--- Test Scenario A: Using Default Parameter Values ---")
        mock_pm_default = MockParamsManager() # No custom params, should use defaults
        analyser_default = ReflectieAnalyser(params_manager=mock_pm_default)

        analysis_result_default = await analyser_default.analyse_reflecties()
        bias_score_default = await analyser_default.calculate_bias_score()
        consistency_result_default = await analyser_default.analyze_reflection_consistency()
        timeframe_bias_result_default = await analyser_default.analyze_timeframe_bias()

        mock_strategy_params = {"id": "DUOAI_Strategy", "parameters": {"emaPeriod": 20, "rsiThreshold": 70}}
        mock_performance_stats = {"winRate": 0.65, "avgProfit": 0.02, "tradeCount": 100}
        duo_ai_bias_default = analysis_result_default.get("biasScores", {}).get("DUOAI_Strategy", {}).get("averageBias", 0.5)

        print(f"DUOAI_Strategy Bias for Default Test: {duo_ai_bias_default:.2f}")
        proposal_default = await analyser_default.generate_mutation_proposal(
            mock_strategy_params, duo_ai_bias_default, mock_performance_stats, timeframe_bias_result_default)
        print("Mutation Proposal (Default Params):")
        print(json.dumps(proposal_default, indent=2))
        # Based on bias 0.59, perf score with defaults: (0.65*50) + min(max(0.02*600, -30),30) + min(100/5, 20) = 32.5 + 12 + 20 = 64.5
        # Bias 0.59 is not >0.7 or <0.3. So adjustment_score = 64.5.
        # 64.5 is not > 70 (adjStrengthenThreshold default) and not < 30 (adjWeakenThreshold default). So action "maintain".
        assert proposal_default['adjustments']['action'] == 'maintain'
        assert abs(proposal_default['currentAdjustmentScore'] - 64.5) < 0.1


        # --- Test Scenario B: Custom Values ---
        print("\n--- Test Scenario B: Using Custom Parameter Values ---")
        custom_params_for_test = {
            "predictAdjust_avgProfitScaleFactor": 800, # Default 600
            "predictAdjust_winRateWeight": 60,         # Default 50
            "predictAdjust_adjStrengthenThreshold": 60, # Default 70 - Lowered to make strengthen easier
            "predictAdjust_paramStrengthenMultiplier": 1.2, # Default 1.1
            "predictAdjust_paramEmaMax": 60 # Default 50
        }
        mock_pm_custom = MockParamsManager(custom_params=custom_params_for_test)
        analyser_custom = ReflectieAnalyser(params_manager=mock_pm_custom)

        # Re-use analysis results as they don't depend on these params
        duo_ai_bias_custom = analysis_result_default.get("biasScores", {}).get("DUOAI_Strategy", {}).get("averageBias", 0.5)

        print(f"DUOAI_Strategy Bias for Custom Test: {duo_ai_bias_custom:.2f}")
        proposal_custom = await analyser_custom.generate_mutation_proposal(
            mock_strategy_params, duo_ai_bias_custom, mock_performance_stats, timeframe_bias_result_default)
        print("Mutation Proposal (Custom Params):")
        print(json.dumps(proposal_custom, indent=2))

        # Recalculate expected score with custom params:
        # scaled_avg_profit_score = min(max(0.02 * 800, -30), 30) = min(max(16, -30), 30) = 16
        # performance_score = (0.65 * 60) + 16 + min(100/5, 20) = 39 + 16 + 20 = 75
        # Bias 0.59 is not >0.7 or <0.3. So adjustment_score = 75.
        # Custom adjStrengthenThreshold = 60. Since 75 > 60, action should be "strengthen".
        assert proposal_custom['adjustments']['action'] == 'strengthen'
        assert abs(proposal_custom['currentAdjustmentScore'] - 75) < 0.1
        # Check if emaPeriod was strengthened using custom multiplier and max
        # Original emaPeriod = 20. New = int(min(20 * 1.2, 60)) = int(min(24, 60)) = 24
        assert proposal_custom['adjustments']['parameterChanges']['emaPeriod'] == 24

        print(f"\nCheck {ANALYSE_LOG} and {BIAS_OUTCOME_LOG} for saved analysis and proposals.")

    asyncio.run(run_test_reflectie_analyser())

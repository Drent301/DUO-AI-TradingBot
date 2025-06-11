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

# Helperfunctie voor JSON-laden (async)
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

async def generate_mutation_proposal(
    strategy: Dict[str, Any],
    bias: float, # Overall bias for the strategy
    performance: Dict[str, Any],
    timeframe_bias_analysis: Optional[Dict[str, Any]] = None # Result from analyze_timeframe_bias
) -> Optional[Dict[str, Any]]:
    """
    Genereer mutatievoorstel met timeframe-data.
    Vertaald van generateMutationProposal in reflectieAnalyser.js.
    """
    if timeframe_bias_analysis is None:
        timeframe_bias_analysis = await analyze_timeframe_bias() # Load all reflections if not provided

    proposal = await predict_strategy_adjustment(strategy, bias, performance)
    if not proposal: return None

    proposal['timeframeSpecificAdjustments'] = {} # Changed key for clarity
    strategy_id_str = strategy['id'] # Ensure it's a string

    if timeframe_bias_analysis: # Ensure it's not None
        for timeframe, strategies_on_tf in timeframe_bias_analysis.items():
            if strategy_id_str in strategies_on_tf:
                tf_bias_info = strategies_on_tf[strategy_id_str]
                # Potentially suggest timeframe-specific parameter tweaks if bias is strong on that TF
                # Example: If 5m timeframe has strong positive bias (tf_bias_info['averageBias'] > 0.7)
                # and overall proposal is to strengthen, maybe make 5m parameters even more aggressive.
                proposal['timeframeSpecificAdjustments'][timeframe] = {
                    "averageBias": tf_bias_info["averageBias"],
                    "confidence": tf_bias_info["confidence"],
                    "suggestedAction": "Monitor" # Placeholder for more detailed TF-specific logic
                }


    proposal['rationale'] = {
        "overallBiasImpact": f"Overall AI bias towards strategy: {bias:.2f}",
        "performanceImpact": f"Win Rate: {performance.get('winRate', 0):.2%}, Avg Profit: {performance.get('avgProfit', 0):.2%}, Trades: {performance.get('tradeCount', 0)}",
        "timeframeConsiderations": f"Timeframe-specific biases: {json.dumps(proposal['timeframeSpecificAdjustments'])}" if proposal['timeframeSpecificAdjustments'] else 'No specific timeframe bias data applied to this proposal.',
        "recommendedAction": proposal.get('adjustments', {}).get('action', 'maintain').capitalize(),
        "confidenceInProposal": proposal.get('confidence', 0.0)
    }

    # Log bias outcome (simplified for now, should be more structured)
    # This is a new log file, so we'll append.
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
    if not isinstance(existing_bias_outcomes, list): existing_bias_outcomes = [] # Ensure it's a list
    existing_bias_outcomes.append(bias_outcome_entry)
    await _write_json_async(BIAS_OUTCOME_LOG, existing_bias_outcomes)


    logger.debug(f'[GenerateMutationProposal] Result: {proposal}')
    return proposal

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import asyncio
    import sys # voor logging in test

    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


    # Mock data voor reflecties
    mock_reflections_data = [
        {
            "token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.8, "combined_bias": 0.7,
            "timeframe": "5m", "trade_context": {"timeframe": "5m", "profit_pct": 0.03}, "timestamp": "2025-06-11T10:00:00Z"
        },
        {
            "token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.6, "combined_bias": 0.4,
             "timeframe": "1h", "trade_context": {"timeframe": "1h", "profit_pct": -0.01}, "timestamp": "2025-06-11T11:00:00Z"
        },
        {
            "token": "BTC/USDT", "strategyId": "Another_Strategy", "combined_confidence": 0.9, "combined_bias": 0.9,
             "timeframe": "15m", "trade_context": {"timeframe": "15m", "profit_pct": 0.05}, "timestamp": "2025-06-11T12:00:00Z"
        },
        {
            "token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.7, "combined_bias": 0.6,
             "timeframe": "5m", "trade_context": {"timeframe": "5m", "profit_pct": 0.01}, "timestamp": "2025-06-11T13:00:00Z"
        },
         { # Invalid reflection example
            "token": "XRP/USDT", "strategyId": "Test_Strategy", "combined_confidence": "high", "combined_bias": "positive",
             "timeframe": "1d", "trade_context": {"timeframe": "1d"}, "timestamp": "2025-06-11T14:00:00Z"
        }
    ]
    # Pre-populate REFLECTIE_LOG for testing functions that load it
    async def setup_mock_log():
        await _write_json_async(REFLECTIE_LOG, mock_reflections_data)
        # Clear other logs for clean test run
        await _write_json_async(ANALYSE_LOG, [])
        await _write_json_async(BIAS_OUTCOME_LOG, [])


    async def run_test_reflectie_analyser():
        await setup_mock_log() # Setup the mock log file

        print("\n--- Test analyse_reflecties (loading from file) ---")
        analysis_result = await analyse_reflecties() # Pass no args to load from file
        print(json.dumps(analysis_result, indent=2))

        print("\n--- Test calculate_bias_score (loading from file) ---")
        bias_score = await calculate_bias_score() # Pass no args
        print(f"Berekende bias score: {bias_score:.2f}")

        print("\n--- Test analyze_reflection_consistency (loading from file) ---")
        consistency_result = await analyze_reflection_consistency() # Pass no args
        print(json.dumps(consistency_result, indent=2))

        print("\n--- Test analyze_timeframe_bias (loading from file) ---")
        timeframe_bias_result = await analyze_timeframe_bias() # Pass no args
        print(json.dumps(timeframe_bias_result, indent=2))

        print("\n--- Test generate_mutation_proposal ---")
        mock_strategy_params = {"id": "DUOAI_Strategy", "parameters": {"emaPeriod": 20, "rsiThreshold": 70}}
        mock_performance_stats = {"winRate": 0.65, "avgProfit": 0.02, "tradeCount": 100}

        # Use an overall bias for the strategy, e.g., from the analysis_result or a specific calculation
        duo_ai_bias = analysis_result.get("biasScores", {}).get("DUOAI_Strategy", {}).get("averageBias", 0.5)


        mutation_proposal_result = await generate_mutation_proposal(
            mock_strategy_params,
            duo_ai_bias,
            mock_performance_stats,
            timeframe_bias_result # Pass the result from analyze_timeframe_bias
        )
        print(json.dumps(mutation_proposal_result, indent=2))

        print(f"\nCheck {ANALYSE_LOG} and {BIAS_OUTCOME_LOG} for saved analysis and proposals.")


    asyncio.run(run_test_reflectie_analyser())

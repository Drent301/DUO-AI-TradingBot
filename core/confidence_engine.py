# core/confidence_engine.py
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import dotenv # Added for __main__ section

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
CONFIDENCE_MEMORY_FILE = os.path.join(MEMORY_DIR, 'confidence_memory.json')
CONFIDENCE_COOLDOWN_SECONDS = 300 # 5 minuten cooldown

os.makedirs(MEMORY_DIR, exist_ok=True)

class ConfidenceEngine:
    """
    Houdt confidence scores bij per reflectie/resultaat en leert van fout/goed scorende combinaties.
    Vertaald van confidenceAdjuster.js en confidenceCycle.js.
    """

    def __init__(self):
        self.confidence_memory = self._load_confidence_memory()
        logger.info(f"ConfidenceEngine geïnitialiseerd. Geheugen: {CONFIDENCE_MEMORY_FILE}")

    def _load_confidence_memory(self) -> Dict[str, Any]:
        try:
            if os.path.exists(CONFIDENCE_MEMORY_FILE) and os.path.getsize(CONFIDENCE_MEMORY_FILE) > 0:
                with open(CONFIDENCE_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {} # Return empty dict if file doesn't exist or is empty
        except (FileNotFoundError, json.JSONDecodeError): # FileNotFoundError for completeness
            logger.warning(f"Confidence geheugenbestand {CONFIDENCE_MEMORY_FILE} niet gevonden of corrupt. Start met leeg geheugen.")
            return {}

    def _save_confidence_memory(self):
        try:
            with open(CONFIDENCE_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.confidence_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan confidence geheugen: {e}")

    def get_confidence_score(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige confidence score op voor een specifieke token/strategie.
        Retourneert een standaardwaarde als niet gevonden (0.5 is neutraal).
        """
        return self.confidence_memory.get(token, {}).get(strategy_id, {}).get('confidence', 0.5)

    def get_max_per_trade_pct(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige max_per_trade_pct op voor een specifieke token/strategie.
        Deze waarde kan de stake amount beïnvloeden.
        """
        return self.confidence_memory.get(token, {}).get(strategy_id, {}).get('max_per_trade_pct', 0.1) # Default 10%


    async def update_confidence(self, token: str, strategy_id: str, ai_reported_confidence: float, trade_result_pct: Optional[float] = None):
        """
        Werkt de confidence score bij op basis van AI-reflectie en trade resultaat.
        Implementeert een cooldown.
        `ai_reported_confidence` is de confidence die de AI zelf rapporteerde (0-1).
        `trade_result_pct` is het winst/verlies percentage van een trade (e.g., 0.05 for +5%).
        """
        if token not in self.confidence_memory:
            self.confidence_memory[token] = {}
        if strategy_id not in self.confidence_memory[token]:
            # Initialize with defaults, including max_per_trade_pct
            self.confidence_memory[token][strategy_id] = {
                'confidence': 0.5,
                'last_update': None,
                'trade_count': 0,
                'total_profit_pct': 0.0,
                'max_per_trade_pct': 0.1 # Initial default max stake percentage
            }


        current_data = self.confidence_memory[token][strategy_id]
        current_learned_confidence = current_data['confidence'] # This is the 'learned' confidence
        last_update_str = current_data.get('last_update')
        last_update_time = datetime.fromisoformat(last_update_str) if last_update_str else None

        # Implementeer cooldown
        if last_update_time and (datetime.now() - last_update_time).total_seconds() < CONFIDENCE_COOLDOWN_SECONDS:
            logger.debug(f"[ConfidenceEngine] Cooldown actief voor {token}/{strategy_id}. Confidence wordt niet bijgewerkt.")
            return

        adjusted_learned_confidence = current_learned_confidence
        max_per_trade_pct = current_data.get('max_per_trade_pct', 0.1) # Huidige max_per_trade_pct

        # Logica van confidenceAdjuster.js
        # ai_reported_confidence is de directe output van een AI model voor een specifieke query.
        # current_learned_confidence is de opgeslagen, geleerde confidence over tijd.

        if trade_result_pct is not None: # Als er een trade resultaat is
            if trade_result_pct > 0: # Winst
                # Verhoog geleerde confidence, sterker als AI ook confident was in zijn (hopelijk correcte) voorspelling
                # en als de winst groot is.
                # De ai_reported_confidence is de confidence van de AI *voor* deze trade (als die beschikbaar was).
                # We nemen aan dat ai_reported_confidence de confidence was van de AI die leidde tot de trade.
                profit_boost = min(trade_result_pct * 5, 1.0) # Max boost van 1.0 bij 20% winst (0.20*5=1.0)
                confidence_increase = (ai_reported_confidence * 0.05) + (profit_boost * 0.1) # Max 0.05 + 0.1 = 0.15
                adjusted_learned_confidence += confidence_increase

                # Verhoog inzet (max_per_trade_pct) als trades goed gaan
                max_per_trade_pct = min(max_per_trade_pct + 0.01 + (profit_boost * 0.02), 0.5) # Max 50% stake
            else: # Verlies
                # Verlaag geleerde confidence, sterker als AI (te) confident was in zijn (foute) voorspelling
                # en als het verlies groot is.
                loss_penalty = min(abs(trade_result_pct) * 5, 1.0) # Max penalty van 1.0 bij 20% verlies
                confidence_decrease = (ai_reported_confidence * 0.05) + (loss_penalty * 0.1) # Max 0.05 + 0.1 = 0.15
                adjusted_learned_confidence -= confidence_decrease

                # Verlaag inzet (max_per_trade_pct) bij verlies
                max_per_trade_pct = max(max_per_trade_pct - 0.01 - (loss_penalty * 0.02), 0.02) # Min 2% stake
        else:
            # Als er geen trade resultaat is (bijv. periodieke reflectie zonder trade),
            # pas de geleerde confidence aan richting de AI's gerapporteerde confidence.
            # De 'ai_reported_confidence' is de output van de AI voor de huidige reflectie.
            # Gewogen gemiddelde:
            adjustment_weight = 0.1 # Hoeveel de nieuwe AI observatie de geleerde confidence beinvloedt
            adjusted_learned_confidence = (current_learned_confidence * (1 - adjustment_weight) + ai_reported_confidence * adjustment_weight)


        adjusted_learned_confidence = max(0.0, min(1.0, adjusted_learned_confidence)) # Houd tussen 0 en 1

        current_data['confidence'] = adjusted_learned_confidence
        current_data['max_per_trade_pct'] = round(max_per_trade_pct, 4) # Afronden voor consistentie
        current_data['last_update'] = datetime.now().isoformat()

        if trade_result_pct is not None: # Alleen updaten als er een trade was
            current_data['trade_count'] = current_data.get('trade_count', 0) + 1
            current_data['total_profit_pct'] = current_data.get('total_profit_pct', 0.0) + trade_result_pct

        await asyncio.to_thread(self._save_confidence_memory)
        logger.info(f"[ConfidenceEngine] Confidence voor {token}/{strategy_id} bijgewerkt naar {adjusted_learned_confidence:.3f}. MaxPerTradePct: {max_per_trade_pct:.3f}.")

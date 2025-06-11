# core/bias_reflector.py
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import dotenv # Added for the __main__ section

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
BIAS_MEMORY_FILE = os.path.join(MEMORY_DIR, 'bias_memory.json')
BIAS_COOLDOWN_SECONDS = 3600 # 1 uur cooldown voor aanpassing

os.makedirs(MEMORY_DIR, exist_ok=True)

class BiasReflector:
    """
    Beheert de lerende bias/voorkeur per strategie/token.
    Vertaald en geoptimaliseerd van biasReflector.js / biasReflectorCooldown.js.
    """

    def __init__(self):
        self.bias_memory = self._load_bias_memory()
        logger.info(f"BiasReflector geïnitialiseerd. Geheugen: {BIAS_MEMORY_FILE}")

    def _load_bias_memory(self) -> Dict[str, Any]:
        try:
            if os.path.exists(BIAS_MEMORY_FILE) and os.path.getsize(BIAS_MEMORY_FILE) > 0:
                with open(BIAS_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {} # Return empty dict if file doesn't exist or is empty
        except (FileNotFoundError, json.JSONDecodeError): # FileNotFoundError for completeness
            logger.warning(f"Bias geheugenbestand {BIAS_MEMORY_FILE} niet gevonden of corrupt. Start met leeg geheugen.")
            return {}

    def _save_bias_memory(self):
        try:
            with open(BIAS_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.bias_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan bias geheugen: {e}")

    def get_bias_score(self, token: str, strategy_id: str) -> float:
        """
        Haalt de huidige bias score op voor een specifieke token/strategie.
        Retourneert een standaardwaarde als niet gevonden (0.5 is neutraal).
        """
        return self.bias_memory.get(token, {}).get(strategy_id, {}).get('bias', 0.5)

    async def update_bias(self, token: str, strategy_id: str, new_ai_bias: float, confidence: float, trade_result_pct: Optional[float] = None):
        """
        Werkt de bias score bij op basis van AI-reflectie en trade resultaat.
        Implementeert een cooldown om te snelle aanpassingen te voorkomen.
        'new_ai_bias' is de bias voorgesteld door de AI (0-1).
        'confidence' is de AI's confidence in die new_ai_bias (0-1).
        'trade_result_pct' is het winst/verlies percentage van een trade (e.g., 0.05 for +5%).
        """
        if token not in self.bias_memory:
            self.bias_memory[token] = {}
        if strategy_id not in self.bias_memory[token]:
            self.bias_memory[token][strategy_id] = {'bias': 0.5, 'last_update': None, 'trade_count': 0, 'total_profit_pct': 0.0}

        current_data = self.bias_memory[token][strategy_id]
        current_bias = current_data['bias']
        last_update_str = current_data.get('last_update')
        last_update_time = datetime.fromisoformat(last_update_str) if last_update_str else None

        # Implementeer cooldown
        if last_update_time and (datetime.now() - last_update_time).total_seconds() < BIAS_COOLDOWN_SECONDS:
            logger.debug(f"[BiasReflector] Cooldown actief voor {token}/{strategy_id}. Bias wordt niet bijgewerkt.")
            return

        # Start met de huidige bias
        calculated_new_bias = current_bias

        # Factor voor hoe sterk een trade resultaat de bias beïnvloedt.
        # Dit kan afhangen van de 'confidence' in de AI en de magnitude van het trade resultaat.
        trade_influence_factor = confidence * 0.1 # Basis invloed, verhoogd door confidence

        if trade_result_pct is not None:
            if trade_result_pct > 0: # Winst
                # Beweeg bias meer richting de AI's voorgestelde bias als die ook positief is,
                # of versterk de huidige bias als de AI neutraal/negatief was maar de trade won.
                # De aanpassing is sterker als de winst groot is.
                # Als AI positief was (new_ai_bias > 0.5), beweeg naar new_ai_bias.
                # Als AI neutraal/negatief was, verhoog de huidige bias een beetje.
                target_bias = new_ai_bias if new_ai_bias > current_bias else current_bias + 0.05 # Kleine boost als AI niet al positiever was
                adjustment = (target_bias - current_bias) * trade_influence_factor * (1 + min(trade_result_pct * 10, 1)) # Max 2x invloed
                calculated_new_bias += adjustment
            else: # Verlies
                # Beweeg bias meer richting de AI's voorgestelde bias als die ook negatief is,
                # of verzwak de huidige bias als de AI neutraal/positief was maar de trade verloor.
                target_bias = new_ai_bias if new_ai_bias < current_bias else current_bias - 0.05 # Kleine verzwakking
                adjustment = (target_bias - current_bias) * trade_influence_factor * (1 + min(abs(trade_result_pct) * 10, 1))
                calculated_new_bias += adjustment
        else:
            # Als er geen trade resultaat is, pas de bias aan richting de AI's suggestie, gewogen door confidence.
            # Dit is een standaard gewogen gemiddelde update.
            calculated_new_bias = (current_bias * (1 - confidence) + new_ai_bias * confidence)

        # Zorg dat bias tussen 0 en 1 blijft
        calculated_new_bias = max(0.0, min(1.0, calculated_new_bias))

        current_data['bias'] = calculated_new_bias
        current_data['last_update'] = datetime.now().isoformat()
        if trade_result_pct is not None: # Alleen trade_count en profit updaten als er een trade was
            current_data['trade_count'] += 1
            current_data['total_profit_pct'] += trade_result_pct

        await asyncio.to_thread(self._save_bias_memory)
        logger.info(f"[BiasReflector] Bias voor {token}/{strategy_id} bijgewerkt naar {calculated_new_bias:.3f} (was {current_bias:.3f}).")

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)
    # Required for StreamHandler in test
    import sys

    async def run_test_bias_reflector():
        # Setup basic logging for the test
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])

        bias_reflector = BiasReflector()
        test_token = "ETH/USDT"
        test_strategy = "DUOAI_Strategy"

        print("\n--- Test BiasReflector ---")

        # Initiele bias
        initial_bias = bias_reflector.get_bias_score(test_token, test_strategy)
        print(f"Initiële bias voor {test_token}/{test_strategy}: {initial_bias:.2f}")

        # Simuleer een winnende trade met positieve AI bias
        print("\nSimuleer winnende trade (+2%), AI bias 0.7, AI confidence 0.8...")
        await bias_reflector.update_bias(test_token, test_strategy, new_ai_bias=0.7, confidence=0.8, trade_result_pct=0.02)
        updated_bias_win = bias_reflector.get_bias_score(test_token, test_strategy)
        print(f"Bias na winnende trade: {updated_bias_win:.3f}")

        # Simuleer een verliezende trade met negatieve AI bias
        print("\nSimuleer verliezende trade (-1%), AI bias 0.3, AI confidence 0.6...")
        await bias_reflector.update_bias(test_token, test_strategy, new_ai_bias=0.3, confidence=0.6, trade_result_pct=-0.01)
        updated_bias_loss = bias_reflector.get_bias_score(test_token, test_strategy)
        print(f"Bias na verliezende trade: {updated_bias_loss:.3f}")

        # Simuleer een update alleen op basis van AI-reflectie (geen trade resultaat)
        print("\nSimuleer update alleen op basis van AI-reflectie (AI bias 0.9, AI confidence 0.9)...")
        await bias_reflector.update_bias(test_token, test_strategy, new_ai_bias=0.9, confidence=0.9)
        updated_bias_ai = bias_reflector.get_bias_score(test_token, test_strategy)
        print(f"Bias na AI-reflectie (geen trade): {updated_bias_ai:.3f}")

        # Test cooldown
        print("\nTest cooldown (probeer direct opnieuw bij te werken)...")
        # Store current time to manually adjust last_update for testing cooldown bypass later
        # For now, this update should be skipped by cooldown
        current_bias_before_cooldown_test = bias_reflector.get_bias_score(test_token, test_strategy)
        await bias_reflector.update_bias(test_token, test_strategy, new_ai_bias=0.1, confidence=0.9, trade_result_pct=-0.05)
        cooldown_bias = bias_reflector.get_bias_score(test_token, test_strategy)
        print(f"Bias tijdens cooldown (zou niet moeten veranderen van {current_bias_before_cooldown_test:.3f}): {cooldown_bias:.3f}")

        # Om de cooldown te testen, zouden we de tijd moeten manipuleren of wachten.
        # Voor een snelle test, kun je handmatig 'last_update' in het JSON-bestand aanpassen naar een oude datum.
        print("\nOm cooldown bypass te testen, pas 'last_update' in bias_memory.json aan of wacht de cooldown periode af.")


    asyncio.run(run_test_bias_reflector())

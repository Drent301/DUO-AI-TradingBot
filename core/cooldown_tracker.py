# core/cooldown_tracker.py
import json
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

# Importeer de ParamsManager om de cooldownDuration op te halen
from core.params_manager import ParamsManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
COOLDOWN_MEMORY_FILE = os.path.join(MEMORY_DIR, 'cooldown_memory.json')

os.makedirs(MEMORY_DIR, exist_ok=True)

class CooldownTracker:
    """
    Beheert AI-specifieke cooldown-periodes per token/strategie.
    Deze cooldowns zijn aanvullend op Freqtrade's ingebouwde cooldowns en
    worden dynamisch ingesteld via de `params_manager`.
    """

    def __init__(self):
        self._cooldown_state = self._load_cooldown_state()
        self.params_manager = ParamsManager() # Instantie om parameters op te halen
        logger.info(f"CooldownTracker geïnitialiseerd. Staat geladen uit: {COOLDOWN_MEMORY_FILE}")

    def _load_cooldown_state(self) -> Dict[str, Any]:
        """Laadt de cooldown staat vanuit een JSON-bestand."""
        try:
            if os.path.exists(COOLDOWN_MEMORY_FILE) and os.path.getsize(COOLDOWN_MEMORY_FILE) > 0:
                with open(COOLDOWN_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Cooldown geheugenbestand {COOLDOWN_MEMORY_FILE} niet gevonden of corrupt. Start met lege staat.")
            return {}

    async def _save_cooldown_state(self):
        """Slaat de huidige cooldown staat op naar een JSON-bestand."""
        try:
            # Ensure the directory exists before trying to save the file
            os.makedirs(os.path.dirname(COOLDOWN_MEMORY_FILE), exist_ok=True)
            await asyncio.to_thread(lambda: json.dump(self._cooldown_state, open(COOLDOWN_MEMORY_FILE, 'w', encoding='utf-8'), indent=2))
        except IOError as e:
            logger.error(f"Fout bij opslaan cooldown staat: {e}")

    async def activate_cooldown(self, token: str, strategy_id: str, reason: str = "general"):
        """
        Activeert een cooldown voor een specifieke token/strategie.
        De duur van de cooldown wordt opgehaald uit `params_manager`.
        """
        cooldown_duration_seconds = self.params_manager.get_param("cooldownDurationSeconds", strategy_id=None) # Globale parameter

        cooldown_end_time = datetime.now() + timedelta(seconds=cooldown_duration_seconds)

        if token not in self._cooldown_state:
            self._cooldown_state[token] = {}

        self._cooldown_state[token][strategy_id] = {
            "end_time": cooldown_end_time.isoformat(),
            "reason": reason,
            "activated_at": datetime.now().isoformat()
        }
        await self._save_cooldown_state()
        logger.info(f"[CooldownTracker] Cooldown geactiveerd voor {token}/{strategy_id} tot {cooldown_end_time.isoformat()} (reden: {reason}).")

    def is_cooldown_active(self, token: str, strategy_id: str) -> bool:
        """
        Controleert of een cooldown actief is voor een specifieke token/strategie.
        """
        if token not in self._cooldown_state or strategy_id not in self._cooldown_state[token]:
            return False

        cooldown_info = self._cooldown_state[token][strategy_id]
        end_time_str = cooldown_info.get("end_time")
        if not end_time_str: return False # Should not happen if well-formed

        cooldown_end_time = datetime.fromisoformat(end_time_str)

        if datetime.now() < cooldown_end_time:
            logger.debug(f"[CooldownTracker] Cooldown actief voor {token}/{strategy_id}. Resterende tijd: {cooldown_end_time - datetime.now()}.")
            return True
        else:
            # Cooldown is voorbij, kan worden opgeschoond (optioneel)
            # del self._cooldown_state[token][strategy_id]
            # self._save_cooldown_state() # Niet elke keer opslaan bij controle, alleen bij wijziging
            return False

    async def deactivate_cooldown(self, token: str, strategy_id: str):
        """
        Deactiveert handmatig een cooldown (voor testen of specifieke scenario's).
        """
        if token in self._cooldown_state and strategy_id in self._cooldown_state[token]:
            del self._cooldown_state[token][strategy_id]
            await self._save_cooldown_state()
            logger.info(f"[CooldownTracker] Cooldown handmatig gedeactiveerd voor {token}/{strategy_id}.")


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import dotenv
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    async def run_test_cooldown_tracker():
        tracker = CooldownTracker()
        test_token = "ETH/USDT"
        test_strategy_id = "DUOAI_Strategy"

        print("--- Test CooldownTracker ---")

        # Test initial state
        print(f"Is cooldown actief voor {test_token}/{test_strategy_id}? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert not tracker.is_cooldown_active(test_token, test_strategy_id)

        # Activate cooldown
        print(f"Activeren cooldown voor {test_token}/{test_strategy_id} (duur: 5 seconden)...")
        # Voor de test, overschrijf de cooldown duration in params_manager
        # This requires params_manager to have a set_param method or similar for testing
        # Assuming params_manager might not have a direct set_param, we'll log this need.
        # For the test to run as is, ParamsManager would need a way to override params.
        # We will proceed assuming params_manager.set_param is available for testing purposes.
        if hasattr(tracker.params_manager, 'set_param'):
            tracker.params_manager.set_param("cooldownDurationSeconds", 5)
        else:
            logger.warning("params_manager.set_param not available. Test may not reflect dynamic cooldown duration.")
            # Fallback: if we can't set it, the test will use the default from params_manager's source

        await tracker.activate_cooldown(test_token, test_strategy_id, reason="test_activation")
        print(f"Is cooldown actief na activatie? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert tracker.is_cooldown_active(test_token, test_strategy_id)

        # Wait for 3 seconds (still in cooldown)
        print("Wachten 3 seconden...")
        await asyncio.sleep(3)
        print(f"Is cooldown actief na 3 seconden? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert tracker.is_cooldown_active(test_token, test_strategy_id)

        # Wait for another 3 seconds (cooldown should be over)
        print("Wachten nogmaals 3 seconden...")
        await asyncio.sleep(3)
        print(f"Is cooldown actief na 6 seconden? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert not tracker.is_cooldown_active(test_token, test_strategy_id)

        # Test handmatige deactivatie
        if hasattr(tracker.params_manager, 'set_param'): # Reset for this test if possible
            tracker.params_manager.set_param("cooldownDurationSeconds", 5)

        await tracker.activate_cooldown(test_token, test_strategy_id, reason="manual_deactivation_test")
        print(f"Cooldown opnieuw geactiveerd. Actief? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert tracker.is_cooldown_active(test_token, test_strategy_id)
        await tracker.deactivate_cooldown(test_token, test_strategy_id)
        print(f"Cooldown gedeactiveerd. Actief? {tracker.is_cooldown_active(test_token, test_strategy_id)}")
        assert not tracker.is_cooldown_active(test_token, test_strategy_id)


    asyncio.run(run_test_cooldown_tracker())

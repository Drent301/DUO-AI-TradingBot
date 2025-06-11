# core/cooldown_tracker.py
import logging
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
COOLDOWN_MEMORY_PATH = os.path.join(MEMORY_DIR, 'cooldown_status.json')

os.makedirs(MEMORY_DIR, exist_ok=True)

# Async JSON read/write helpers (similar to those in other modules)
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
        return {}

async def _write_json_async(filepath: str, data: Dict[str, Any]):
    def write_file_sync():
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    try:
        await asyncio.to_thread(write_file_sync)
    except Exception as e:
        logger.error(f"Error writing to JSON file {filepath}: {e}")


class CooldownTracker:
    """
    Beheert cooldown periodes voor AI-acties om overtrading en snelle fluctuaties te voorkomen.
    Vertaald van concepten in cooldownManager.js en biasReflector.js (voor cooldown aspect).
    """

    def __init__(self, default_cooldown_minutes: int = 30):
        self.default_cooldown_minutes = default_cooldown_minutes
        # In-memory cache voor snelle checks, geladen vanuit JSON bij initialisatie (of later)
        self.cooldown_status: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False # Om laden van memory te beheren
        logger.info(f"CooldownTracker geïnitialiseerd met default cooldown: {default_cooldown_minutes} min.")

    async def _load_memory(self):
        """Laadt cooldown status uit JSON bestand."""
        if not self.is_initialized:
            self.cooldown_status = await _read_json_async(COOLDOWN_MEMORY_PATH)
            self.is_initialized = True
            logger.debug("Cooldown status geladen vanuit JSON.")

    async def _save_memory(self):
        """Slaat huidige cooldown status op naar JSON."""
        await _write_json_async(COOLDOWN_MEMORY_PATH, self.cooldown_status)
        logger.debug("Cooldown status opgeslagen naar JSON.")

    def _get_cooldown_key(self, symbol: str, strategy_id: str, action_type: str = "general_ai") -> str:
        """Creëert een unieke key voor de cooldown entry."""
        return f"{symbol}_{strategy_id}_{action_type}"

    async def set_cooldown(self, symbol: str, strategy_id: str, action_type: str = "general_ai", minutes: Optional[int] = None):
        """
        Activeert een cooldown periode voor een specifieke combinatie van symbool, strategie en actie.
        """
        await self._load_memory()
        cooldown_minutes = minutes if minutes is not None else self.default_cooldown_minutes
        key = self._get_cooldown_key(symbol, strategy_id, action_type)

        cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=cooldown_minutes)

        if key not in self.cooldown_status:
            self.cooldown_status[key] = {}

        self.cooldown_status[key]['cooldown_until'] = cooldown_until.isoformat()
        self.cooldown_status[key]['last_set'] = datetime.now(timezone.utc).isoformat()
        self.cooldown_status[key]['duration_minutes'] = cooldown_minutes

        await self._save_memory()
        logger.info(f"Cooldown geactiveerd voor '{key}' tot {cooldown_until.isoformat()}.")

    async def is_cooldown_active(self, symbol: str, strategy_id: str, action_type: str = "general_ai") -> bool:
        """
        Controleert of er een actieve cooldown is voor de gegeven combinatie.
        """
        await self._load_memory()
        key = self._get_cooldown_key(symbol, strategy_id, action_type)

        if key in self.cooldown_status:
            cooldown_entry = self.cooldown_status[key]
            if 'cooldown_until' in cooldown_entry:
                try:
                    cooldown_until_dt = datetime.fromisoformat(cooldown_entry['cooldown_until'])
                    if cooldown_until_dt.tzinfo is None: # Als er geen timezone info is, neem UTC aan
                        cooldown_until_dt = cooldown_until_dt.replace(tzinfo=timezone.utc)

                    now_utc = datetime.now(timezone.utc)

                    if now_utc < cooldown_until_dt:
                        logger.debug(f"Cooldown actief voor '{key}'. Resterende tijd: {cooldown_until_dt - now_utc}")
                        return True
                    else:
                        logger.debug(f"Cooldown voor '{key}' is verlopen op {cooldown_until_dt.isoformat()}.")
                        # Optioneel: verwijder verlopen cooldowns uit memory
                        # del self.cooldown_status[key] # Kan leiden tot veel schrijfoperaties
                        return False
                except ValueError:
                    logger.warning(f"Ongeldige datumformaat in cooldown memory voor key '{key}': {cooldown_entry['cooldown_until']}")
                    return False # Behandel als geen cooldown bij corrupte data

        logger.debug(f"Geen actieve cooldown gevonden voor '{key}'.")
        return False

    async def clear_cooldown(self, symbol: str, strategy_id: str, action_type: str = "general_ai"):
        """Verwijdert een specifieke cooldown."""
        await self._load_memory()
        key = self._get_cooldown_key(symbol, strategy_id, action_type)
        if key in self.cooldown_status:
            del self.cooldown_status[key]
            await self._save_memory()
            logger.info(f"Cooldown voor '{key}' handmatig verwijderd.")
        else:
            logger.info(f"Geen cooldown om te verwijderen voor '{key}'.")

    async def get_cooldown_details(self, symbol: str, strategy_id: str, action_type: str = "general_ai") -> Optional[Dict[str, Any]]:
        """Haalt details op van een actieve cooldown, indien aanwezig."""
        await self._load_memory()
        key = self._get_cooldown_key(symbol, strategy_id, action_type)
        if await self.is_cooldown_active(symbol, strategy_id, action_type): # Re-check to ensure it's still active
            return self.cooldown_status.get(key)
        return None

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    async def run_test_cooldown_tracker():
        tracker = CooldownTracker(default_cooldown_minutes=1) # Korte cooldown voor test

        test_symbol = "BTC/USDT"
        test_strategy = "TestStrategy"
        test_action = "entry_signal"

        print(f"\n--- Test CooldownTracker (default: {tracker.default_cooldown_minutes} min) ---")

        # 1. Check initieel: geen cooldown
        is_active = await tracker.is_cooldown_active(test_symbol, test_strategy, test_action)
        print(f"Initieel cooldown actief voor '{test_action}': {is_active}")
        assert not is_active

        # 2. Zet cooldown
        await tracker.set_cooldown(test_symbol, test_strategy, test_action, minutes=1) # 1 minuut
        is_active_after_set = await tracker.is_cooldown_active(test_symbol, test_strategy, test_action)
        print(f"Cooldown actief na zetten voor '{test_action}': {is_active_after_set}")
        assert is_active_after_set

        details = await tracker.get_cooldown_details(test_symbol, test_strategy, test_action)
        print(f"Details na zetten: {details}")
        assert details is not None
        assert details['duration_minutes'] == 1

        # 3. Test met andere action_type (zou geen cooldown moeten hebben)
        other_action = "exit_signal"
        is_active_other = await tracker.is_cooldown_active(test_symbol, test_strategy, other_action)
        print(f"Cooldown actief voor '{other_action}': {is_active_other}")
        assert not is_active_other

        # 4. Wacht tot cooldown verloopt (voor testen, normaal zou dit langer zijn)
        print("Wachten tot cooldown (1 minuut) verloopt...")
        # In een echte test zou je `asyncio.sleep` mocken of een test-specifieke clock gebruiken.
        # Hier simuleren we het passeren van tijd door kort te slapen.
        # Voor een 1-minuut cooldown, slapen we iets langer om zeker te zijn.
        # await asyncio.sleep(65) # Te lang voor een snelle unit test
        # Manier om dit te testen zonder echt te wachten:
        # Forceer de cooldown_until tijd in het verleden voor de test (vereist aanpassing van _load_memory of interne structuur)
        # Of, voor deze test, verminderen we de cooldown en wachten we kort.

        # Voor snelle test, zet een zeer korte cooldown en wacht
        await tracker.set_cooldown(test_symbol, test_strategy, "short_test", minutes=0.02) # 0.02 min = 1.2 sec
        print(f"Test met korte cooldown (1.2s) voor 'short_test'...")
        await asyncio.sleep(1.5)
        is_active_short_expired = await tracker.is_cooldown_active(test_symbol, test_strategy, "short_test")
        print(f"Cooldown 'short_test' actief na 1.5s: {is_active_short_expired}")
        assert not is_active_short_expired


        # 5. Verwijder cooldown handmatig
        await tracker.set_cooldown(test_symbol, test_strategy, "manual_clear_test", minutes=5)
        is_active_before_clear = await tracker.is_cooldown_active(test_symbol, test_strategy, "manual_clear_test")
        assert is_active_before_clear, "Cooldown should be active before manual clear"
        await tracker.clear_cooldown(test_symbol, test_strategy, "manual_clear_test")
        is_active_after_clear = await tracker.is_cooldown_active(test_symbol, test_strategy, "manual_clear_test")
        print(f"Cooldown 'manual_clear_test' actief na wissen: {is_active_after_clear}")
        assert not is_active_after_clear

        print("\nCooldownTracker tests voltooid.")
        # Optioneel: ruim het testgeheugenbestand op
        if os.path.exists(COOLDOWN_MEMORY_PATH):
            os.remove(COOLDOWN_MEMORY_PATH)

    asyncio.run(run_test_cooldown_tracker())

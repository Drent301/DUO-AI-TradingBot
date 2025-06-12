# core/params_manager.py
import json
import logging
import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
PARAMS_FILE = os.path.join(MEMORY_DIR, 'learned_params.json')

os.makedirs(MEMORY_DIR, exist_ok=True)

class ParamsManager:
    """
    Beheert het laden, opslaan en dynamisch bijwerken van leerbare variabelen.
    Deze variabelen worden door AI-modules bijgesteld en door de Freqtrade-strategie gebruikt.
    """

    def __init__(self):
        self._params: Dict[str, Any] = self._load_params()
        logger.info(f"ParamsManager geïnitialiseerd. Parameters geladen uit: {PARAMS_FILE}")

    def _load_params(self) -> Dict[str, Any]:
        """Laadt de leerbare parameters uit een JSON-bestand."""
        try:
            if os.path.exists(PARAMS_FILE) and os.path.getsize(PARAMS_FILE) > 0:
                with open(PARAMS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Parameterbestand {PARAMS_FILE} niet gevonden of is leeg. Start met standaardparameters.")
                return self._get_default_params()
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Parameterbestand {PARAMS_FILE} niet gevonden of corrupt. Start met standaardparameters.")
            return self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Retourneert een dictionary met standaard (begin)parameters."""
        return {
            "global": {
                "maxTradeRiskPct": 0.02,
                "slippageTolerancePct": 0.001,
                "cooldownDurationSeconds": 300
                # Add other global defaults as needed
            },
            "strategies": {
                "DUOAI_Strategy": { # Example strategy
                    "entryConvictionThreshold": 0.7,
                    "exitConvictionDropTrigger": 0.4,
                    "cnnPatternWeight": 1.0,
                    "strongPatternThreshold": 0.5,
                    "preferredPairs": [],
                    "minimal_roi": {"0": 0.05, "30": 0.03, "60": 0.02, "120": 0.01},
                    "stoploss": -0.10,
                    "trailing_stop_positive": 0.005,
                    "trailing_stop_positive_offset": 0.01
                    # Add other strategy defaults as needed
                }
            },
            "timeOfDayEffectiveness": {}
            # Add other top-level default sections as needed
        }

    async def _save_params(self):
        """Slaat de huidige leerbare parameters op naar een JSON-bestand."""
        try:
            # Using a lambda to pass arguments to json.dump correctly with to_thread
            # Ensure the lambda correctly opens the file for writing
            def _dump_json():
                with open(PARAMS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(self._params, f, indent=4) # Changed indent to 4

            await asyncio.to_thread(_dump_json)
        except IOError as e:
            logger.error(f"Fout bij opslaan leerbare parameters: {e}")

    def get_param(self, key: str, strategy_id: Optional[str] = None, default: Any = None) -> Any:
        """Haalt een specifieke leerbare parameter op, met fallback naar defaults en een uiteindelijke default waarde."""
        default_params_snapshot = self._get_default_params() # Get a fresh copy of defaults

        # 1. Try live strategy-specific parameters
        if strategy_id:
            strategy_params = self._params.get("strategies", {}).get(strategy_id, {})
            if key in strategy_params:
                return strategy_params[key]

        # 2. Try live root-level parameters (e.g., timeOfDayEffectiveness)
        #    Exclude "global" and "strategies" themselves from this type of lookup.
        if key in self._params and key not in ["global", "strategies"]:
            return self._params[key]

        # 3. Try live global parameters
        if key in self._params.get("global", {}):
            return self._params["global"][key]

        # If not found in live parameters, try default parameters with the same lookup order
        # 4. Try default strategy-specific parameters
        if strategy_id:
            default_strategy_params = default_params_snapshot.get("strategies", {}).get(strategy_id, {})
            if key in default_strategy_params:
                return default_strategy_params[key]

        # 5. Try default root-level parameters
        if key in default_params_snapshot and key not in ["global", "strategies"]:
            return default_params_snapshot[key]

        # 6. Try default global parameters
        if key in default_params_snapshot.get("global", {}):
            return default_params_snapshot["global"][key]

        # 7. If not found anywhere, return the provided default argument
        return default

    async def set_param(self, key: str, value: Any, strategy_id: Optional[str] = None):
        """Stelt een leerbare parameter in en slaat op."""
        default_params_snapshot = self._get_default_params() # To understand structure

        if strategy_id:
            if "strategies" not in self._params: self._params["strategies"] = {}
            if strategy_id not in self._params["strategies"]: self._params["strategies"][strategy_id] = {} # Initialize new strategy dict
            self._params["strategies"][strategy_id][key] = value
            logger.info(f"Leerbare parameter '{key}' voor strategie '{strategy_id}' bijgewerkt naar: {value}")
        # Check if the key is a known top-level default key (like timeOfDayEffectiveness)
        elif key in default_params_snapshot and key not in ["global", "strategies"]:
             self._params[key] = value
             logger.info(f"Root-level leerbare parameter '{key}' bijgewerkt naar: {value}")
        else: # Assume global parameter otherwise
            if "global" not in self._params: self._params["global"] = {}
            self._params["global"][key] = value
            logger.info(f"Globale leerbare parameter '{key}' bijgewerkt naar: {value}")
        await self._save_params()

    async def update_strategy_roi_sl_params(self, strategy_id: str, new_roi: Dict[str, float], new_stoploss: float, new_trailing_stop_positive: float, new_trailing_stop_positive_offset: float):
        """Werkt ROI, Stoploss en Trailing Stop parameters voor een strategie bij."""
        if "strategies" not in self._params: self._params["strategies"] = {}
        if strategy_id not in self._params["strategies"]: self._params["strategies"][strategy_id] = {}

        self._params["strategies"][strategy_id]["minimal_roi"] = new_roi
        self._params["strategies"][strategy_id]["stoploss"] = new_stoploss
        self._params["strategies"][strategy_id]["trailing_stop_positive"] = new_trailing_stop_positive
        self._params["strategies"][strategy_id]["trailing_stop_positive_offset"] = new_trailing_stop_positive_offset
        self._params["strategies"][strategy_id]["last_mutated"] = datetime.now().isoformat()

        await self._save_params()
        logger.info(f"ROI/SL parameters voor strategie '{strategy_id}' bijgewerkt.")

    async def update_time_effectiveness(self, data: Dict[int, float]): # Note: Test suite uses str keys for hour
        """Werkt tijd-van-dag effectiviteit data bij."""
        # Convert integer keys from older versions if necessary, ensure string keys for JSON
        self._params["timeOfDayEffectiveness"] = {str(k): v for k, v in data.items()}
        await self._save_params()
        logger.info("Tijd-van-dag effectiviteit bijgewerkt.")


# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    import dotenv
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    async def run_test_params_manager():
        # Clear the params file for a clean test run
        if os.path.exists(PARAMS_FILE):
            os.remove(PARAMS_FILE)
            print(f"Verwijderde bestaand parameterbestand: {PARAMS_FILE}")

        params_manager = ParamsManager()
        test_strategy_id = "DUOAI_Strategy"
        other_strategy_id = "OTHER_Strategy" # For testing fallback

        print("--- Test ParamsManager ---")

        # Get default global parameter
        max_risk = params_manager.get_param("maxTradeRiskPct")
        print(f"Standaard maxTradeRiskPct (global): {max_risk}")
        assert max_risk == 0.02

        # Test get_param with explicit default
        non_existent = params_manager.get_param("nonExistentKey", default="fallback_value")
        print(f"Test nonExistentKey met default: {non_existent}")
        assert non_existent == "fallback_value"


        # Set a global parameter
        await params_manager.set_param("maxTradeRiskPct", 0.03)
        print(f"Nieuwe maxTradeRiskPct (global): {params_manager.get_param('maxTradeRiskPct')}")
        assert params_manager.get_param("maxTradeRiskPct") == 0.03

        # Test fetching timeOfDayEffectiveness (should be at root)
        tod_effectiveness_default = params_manager.get_param("timeOfDayEffectiveness")
        print(f"Standaard timeOfDayEffectiveness: {tod_effectiveness_default}")
        assert tod_effectiveness_default == {}


        # Get strategy-specific parameter (cnnPatternWeight from default)
        cnn_weight_default = params_manager.get_param("cnnPatternWeight", test_strategy_id)
        print(f"Standaard cnnPatternWeight voor {test_strategy_id}: {cnn_weight_default}")
        assert cnn_weight_default == 1.0

        strong_pattern_threshold = params_manager.get_param("strongPatternThreshold", test_strategy_id)
        print(f"Standaard strongPatternThreshold voor {test_strategy_id}: {strong_pattern_threshold}")
        assert strong_pattern_threshold == 0.5

        # Test getting a param for a strategy not in current _params, but defined in defaults
        default_only_strat_param = params_manager.get_param("entryConvictionThreshold", "NEW_DEFAULT_STRAT_ONLY", default=0.99)
        # Assuming NEW_DEFAULT_STRAT_ONLY is not in _get_default_params, it should return the 'default' argument
        # If NEW_DEFAULT_STRAT_ONLY *was* in defaults, it would return that. Let's assume it's not for this test.
        # To make this test more robust, we'd ensure NEW_DEFAULT_STRAT_ONLY isn't in _get_default_params
        # For now, if it's not, it will correctly fall back to the provided default=0.99
        # If it *is* in defaults (e.g. if DUOAI_Strategy was used), it would return that default (0.7)
        # Let's use a key that's unlikely to be in DUOAI_Strategy defaults:
        unique_key_for_new_strat = params_manager.get_param("uniqueKeyForNewStrat", "NEW_DEFAULT_STRAT_ONLY", default="new_strat_fallback")
        print(f"Test uniqueKeyForNewStrat voor NEW_DEFAULT_STRAT_ONLY: {unique_key_for_new_strat}")
        assert unique_key_for_new_strat == "new_strat_fallback"


        # Get a parameter that only exists in default strategy for a different strategy_id (should fallback to None without explicit default)
        cnn_weight_other_fallback = params_manager.get_param("cnnPatternWeight", other_strategy_id)
        print(f"Standaard cnnPatternWeight voor {other_strategy_id} (geen default, niet global): {cnn_weight_other_fallback}")
        # This assertion depends on OTHER_Strategy NOT being in _get_default_params(). If it is, this would fail.
        # If OTHER_Strategy has no "cnnPatternWeight" in defaults, it will be None.
        assert cnn_weight_other_fallback is None


        # Set a strategy-specific parameter
        await params_manager.set_param("entryConvictionThreshold", 0.8, test_strategy_id)
        print(f"Nieuwe entryConvictionThreshold voor {test_strategy_id}: {params_manager.get_param('entryConvictionThreshold', test_strategy_id)}")
        assert params_manager.get_param("entryConvictionThreshold", test_strategy_id) == 0.8

        # Set cnnPatternWeight for the other strategy
        await params_manager.set_param("cnnPatternWeight", 0.75, other_strategy_id)
        print(f"Nieuwe cnnPatternWeight voor {other_strategy_id}: {params_manager.get_param('cnnPatternWeight', other_strategy_id)}")
        assert params_manager.get_param("cnnPatternWeight", other_strategy_id) == 0.75


        # Test ROI/SL update
        new_roi = {"0": 0.06, "60": 0.03}
        new_sl = -0.12
        new_tsp = 0.008
        new_tspo = 0.015
        await params_manager.update_strategy_roi_sl_params(test_strategy_id, new_roi, new_sl, new_tsp, new_tspo)

        updated_strategy_params = params_manager.get_param("minimal_roi", test_strategy_id)
        print(f"Updated ROI for {test_strategy_id}: {updated_strategy_params}")
        assert updated_strategy_params == new_roi

        # Test time effectiveness update
        time_data = {0: 0.1, 1: 0.05, 2: -0.02} # Test with int keys
        time_data_str_keys = {str(k): v for k,v in time_data.items()} # Expected after processing
        await params_manager.update_time_effectiveness(time_data) # Pass int keys
        fetched_time_effectiveness = params_manager.get_param('timeOfDayEffectiveness')
        print(f"Tijd-van-dag effectiviteit: {fetched_time_effectiveness}")
        assert fetched_time_effectiveness == time_data_str_keys # Assert against str keys

        # Verify loaded state
        print("\nVerifiëren van opgeslagen staat (door opnieuw te laden)...")
        reloaded_manager = ParamsManager()
        print(f"Hergeladen maxTradeRiskPct: {reloaded_manager.get_param('maxTradeRiskPct')}")
        assert reloaded_manager.get_param("maxTradeRiskPct") == 0.03
        print(f"Hergeladen entryConvictionThreshold voor {test_strategy_id}: {reloaded_manager.get_param('entryConvictionThreshold', test_strategy_id)}")
        assert reloaded_manager.get_param("entryConvictionThreshold", test_strategy_id) == 0.8

        # cnnPatternWeight for test_strategy_id was not explicitly set for test_strategy_id, so it should load default.
        print(f"Hergeladen cnnPatternWeight voor {test_strategy_id}: {reloaded_manager.get_param('cnnPatternWeight', test_strategy_id)}")
        assert reloaded_manager.get_param("cnnPatternWeight", test_strategy_id) == 1.0

        # Check the cnnPatternWeight for other_strategy_id after reload
        print(f"Hergeladen cnnPatternWeight voor {other_strategy_id}: {reloaded_manager.get_param('cnnPatternWeight', other_strategy_id)}")
        assert reloaded_manager.get_param("cnnPatternWeight", other_strategy_id) == 0.75

        reloaded_strong_threshold = reloaded_manager.get_param('strongPatternThreshold', test_strategy_id)
        print(f"Hergeladen strongPatternThreshold voor {test_strategy_id}: {reloaded_strong_threshold}")
        assert reloaded_strong_threshold == 0.5

        print(f"Hergeladen ROI voor {test_strategy_id}: {reloaded_manager.get_param('minimal_roi', test_strategy_id)}")
        assert reloaded_manager.get_param("minimal_roi", test_strategy_id) == new_roi

        fetched_time_effectiveness_reloaded = reloaded_manager.get_param('timeOfDayEffectiveness')
        print(f"Hergeladen Tijd-van-dag effectiviteit: {fetched_time_effectiveness_reloaded}")
        assert fetched_time_effectiveness_reloaded == time_data_str_keys

        print("\nParamsManager tests succesvol afgerond.")

    asyncio.run(run_test_params_manager())

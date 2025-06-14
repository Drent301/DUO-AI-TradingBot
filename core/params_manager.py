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
                "cooldownDurationSeconds": 300,
                "data_fetch_start_date_str": "2020-01-01",
                "data_fetch_end_date_str": "",
                "data_fetch_pairs": ["ETH/USDT", "ETH/BTC", "LSK/BTC", "ZEN/BTC", "ETH/EUR"],
                "data_fetch_timeframes": ["1h", "4h"],
                "patterns_to_train": ["bullFlag", "bearishEngulfing"],
                "gold_standard_data_path": "", # Path to gold standard data CSVs
                "pattern_labeling_configs": {
                    "bullFlag": {
                        "future_N_candles": 20,
                        "profit_threshold_pct": 0.02,
                        "loss_threshold_pct": -0.01
                    },
                    "bearishEngulfing": {
                        "future_N_candles": 20,
                        "profit_threshold_pct": 0.02,
                        "loss_threshold_pct": -0.01
                    }
                },
                "current_cnn_architecture_key": "default_simple",
                "cnn_architecture_configs": {
                    "default_simple": {
                        "num_conv_layers": 2,
                        "filters_per_layer": [16, 32],
                        "kernel_sizes_per_layer": [3, 3],
                        "strides_per_layer": [1, 1],
                        "padding_per_layer": [1, 1],
                        "pooling_types_per_layer": ["max", "max"],
                        "pooling_kernel_sizes_per_layer": [2, 2],
                        "pooling_strides_per_layer": [2, 2],
                        "use_batch_norm": False,
                        "dropout_rate": 0.0
                    },
                    "deeper_with_batchnorm": {
                        "num_conv_layers": 3,
                        "filters_per_layer": [16, 32, 64],
                        "kernel_sizes_per_layer": [3, 3, 3],
                        "strides_per_layer": [1, 1, 1],
                        "padding_per_layer": [1, 1, 1],
                        "pooling_types_per_layer": ["max", "max", "max"],
                        "pooling_kernel_sizes_per_layer": [2, 2, 2],
                        "pooling_strides_per_layer": [2, 2, 2],
                        "use_batch_norm": True,
                        "dropout_rate": 0.25
                    }
                },
                "perform_cross_validation": True,
                "cv_num_splits": 5,
                "perform_backtesting": True,
                "backtest_start_date_str": "2023-06-01",
                "backtest_entry_threshold": 0.7,
                "backtest_take_profit_pct": 0.05,
                "backtest_stop_loss_pct": 0.02,
                "backtest_hold_duration_candles": 20,
                "backtest_initial_capital": 1000.0,
                "backtest_stake_pct_capital": 0.1,
                "default_strategy_id": "DefaultPipelineRunStrategy", # Added default strategy ID

                # Hyperparameter Optimization (HPO) settings
                "perform_hyperparameter_optimization": False, # Whether to run HPO
                "hpo_num_trials": 20, # Number of HPO trials to run (e.g., 10-50)
                "hpo_sampler": "TPE",   # Optuna sampler: 'TPE', 'Random'
                "hpo_pruner": "Median", # Optuna pruner: 'Median', 'None'
                "hpo_metric_to_optimize": "val_loss", # Metric to optimize: 'val_loss', 'val_accuracy', 'val_f1', 'val_auc'
                "hpo_direction_to_optimize": "minimize", # Direction: 'minimize' for loss, 'maximize' for acc/f1/auc
                "regimes_to_train": ["all"] # List of market regimes to train models for (e.g., ["bull", "bear", "all"])
            },
            "strategies": {
                "DUOAI_Strategy": {
                    "entryConvictionThreshold": 0.7,
                    "exitConvictionDropTrigger": 0.4,
                    "cnnPatternWeight": 1.0, # Fallback weight
                    "cnn_bullFlag_weight": 1.0, # Specific weight for bullFlag
                    "cnn_bearishEngulfing_weight": 1.0, # Specific weight for bearishEngulfing
                    "strongPatternThreshold": 0.5, # Default threshold for strong patterns
                    "entryRulePatternScore": 0.7, # NIEUW: Default score for a detected rule-based entry pattern
                    "exitRulePatternScore": 0.7,  # NIEUW: Default score for a detected rule-based exit pattern
                    "preferredPairs": [],
                    "minimal_roi": {"0": 0.05, "30": 0.03, "60": 0.02, "120": 0.01},
                    "stoploss": -0.10,
                    "trailing_stop_positive": 0.005,
                    "trailing_stop_positive_offset": 0.01
                },
                "DefaultPipelineRunStrategy": { # Added an entry for the default strategy
                    "entryConvictionThreshold": 0.6,
                    "exitConvictionDropTrigger": 0.3,
                    "cnnPatternWeight": 1.0, # Fallback weight
                    "cnn_bullFlag_weight": 1.0, # Specific weight for bullFlag
                    "cnn_bearishEngulfing_weight": 1.0, # Specific weight for bearishEngulfing
                    "strongPatternThreshold": 0.6,
                    "entryRulePatternScore": 0.6,
                    "exitRulePatternScore": 0.6,
                    "preferredPairs": [],
                    "minimal_roi": {"0": 0.04, "30": 0.02, "60": 0.01},
                    "stoploss": -0.08,
                    "trailing_stop_positive": 0.004,
                    "trailing_stop_positive_offset": 0.008
                }
            },
            "timeOfDayEffectiveness": {}
        }

    async def _save_params(self):
        """Slaat de huidige leerbare parameters op naar een JSON-bestand."""
        try:
            await asyncio.to_thread(lambda: json.dump(self._params, open(PARAMS_FILE, 'w', encoding='utf-8'), indent=2))
        except IOError as e:
            logger.error(f"Fout bij opslaan leerbare parameters: {e}")

    def get_param(self, key: str, strategy_id: Optional[str] = None) -> Any:
        """Haalt een specifieke leerbare parameter op."""
        # Eerst proberen strategie-specifieke parameter op te halen
        if strategy_id and "strategies" in self._params and strategy_id in self._params["strategies"]:
            if key in self._params["strategies"][strategy_id]:
                return self._params["strategies"][strategy_id][key]

        # Fallback naar globale parameters als strategie-specifiek niet gevonden of niet van toepassing
        if key in self._params.get("global", {}): # Ensure "global" key exists
            return self._params["global"][key]

        # Check root level for keys like timeOfDayEffectiveness that might not be under "global"
        if key in self._params and key not in ["strategies", "global"]:
            return self._params[key]


        # Als niets gevonden, haal op uit default params
        default_global = self._get_default_params()["global"].get(key)
        if default_global is not None: return default_global

        default_strategy_params = self._get_default_params()["strategies"].get(strategy_id, {})
        default_strategy_value = default_strategy_params.get(key)
        if default_strategy_value is not None: return default_strategy_value

        # Fallback for timeOfDayEffectiveness if not in global or strategy specific
        if key == "timeOfDayEffectiveness" and key in self._get_default_params():
            return self._get_default_params()[key]

        return None # Return None if parameter is not found anywhere

    async def set_param(self, key: str, value: Any, strategy_id: Optional[str] = None):
        """Stelt een leerbare parameter in en slaat op."""
        if strategy_id:
            if "strategies" not in self._params: self._params["strategies"] = {}
            if strategy_id not in self._params["strategies"]: self._params["strategies"][strategy_id] = {}
            self._params["strategies"][strategy_id][key] = value
            logger.info(f"Leerbare parameter '{key}' voor strategie '{strategy_id}' bijgewerkt naar: {value}")
        else:
            # Check if the key is a known global key or if it's 'timeOfDayEffectiveness'
            if key in self._get_default_params()["global"]:
                 self._params["global"][key] = value
            elif key == "timeOfDayEffectiveness" or key in self._get_default_params(): # check root level default keys
                 self._params[key] = value # Store at root level for global settings like timeOfDayEffectiveness
            else:
                 # If it's an unknown key and no strategy_id, default to storing in "global"
                 if "global" not in self._params: self._params["global"] = {}
                 self._params["global"][key] = value
            logger.info(f"Globale leerbare parameter '{key}' bijgewerkt naar: {value}")
        await self._save_params()

    async def update_strategy_roi_sl_params(self,
                                        strategy_id: str,
                                        new_roi: Optional[Dict[str, float]] = None, # Made optional for flexibility
                                        new_stoploss: Optional[float] = None, # Made optional for flexibility
                                        new_trailing_stop: Optional[float] = None,
                                        new_trailing_only_offset_is_reached: Optional[float] = None):
        """Werkt ROI, Stoploss en Trailing Stop parameters voor een strategie bij."""
        if "strategies" not in self._params: self._params["strategies"] = {}
        if strategy_id not in self._params["strategies"]: self._params["strategies"][strategy_id] = {}

        changes_made = False
        if new_roi is not None:
            self._params["strategies"][strategy_id]["minimal_roi"] = new_roi
            changes_made = True
        if new_stoploss is not None:
            self._params["strategies"][strategy_id]["stoploss"] = new_stoploss
            changes_made = True
        if new_trailing_stop is not None:
            self._params["strategies"][strategy_id]["trailing_stop_positive"] = new_trailing_stop
            changes_made = True
        if new_trailing_only_offset_is_reached is not None:
            self._params["strategies"][strategy_id]["trailing_stop_positive_offset"] = new_trailing_only_offset_is_reached
            changes_made = True

        if changes_made:
            self._params["strategies"][strategy_id]["last_mutated"] = datetime.now().isoformat()
            await self._save_params()
            logger.info(f"ROI/SL/Trailing parameters voor strategie '{strategy_id}' bijgewerkt.")
        else:
            logger.info(f"Geen ROI/SL/Trailing parameters om bij te werken voor strategie '{strategy_id}'.")

    async def update_time_effectiveness(self, data: Dict[int, float]):
        """Werkt tijd-van-dag effectiviteit data bij."""
        # Global parameter, stored at the root of the params structure
        self._params["timeOfDayEffectiveness"] = data
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

        # Set a global parameter
        await params_manager.set_param("maxTradeRiskPct", 0.03) # Implicitly global due to default structure
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

        entry_rule_score = params_manager.get_param("entryRulePatternScore", test_strategy_id)
        print(f"Standaard entryRulePatternScore voor {test_strategy_id}: {entry_rule_score}")
        assert entry_rule_score == 0.7

        exit_rule_score = params_manager.get_param("exitRulePatternScore", test_strategy_id)
        print(f"Standaard exitRulePatternScore voor {test_strategy_id}: {exit_rule_score}")
        assert exit_rule_score == 0.7

        # Get a parameter that only exists in default strategy for a different strategy_id (should fallback to None)
        cnn_weight_other_fallback = params_manager.get_param("cnnPatternWeight", other_strategy_id)
        print(f"Standaard cnnPatternWeight voor {other_strategy_id} (geen default, niet global): {cnn_weight_other_fallback}")
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
        # Update the call to update_strategy_roi_sl_params to reflect the new signature
        # Old call: await params_manager.update_strategy_roi_sl_params(test_strategy_id, new_roi, new_sl, new_tsp, new_tspo)
        await params_manager.update_strategy_roi_sl_params(strategy_id=test_strategy_id,
                                                            new_roi=new_roi,
                                                            new_stoploss=new_sl,
                                                            new_trailing_stop=0.008,  # Example value
                                                            new_trailing_only_offset_is_reached=0.015) # Example value

        updated_strategy_params = params_manager.get_param("minimal_roi", test_strategy_id)
        print(f"Updated ROI for {test_strategy_id}: {updated_strategy_params}")
        assert updated_strategy_params == new_roi

        # Test time effectiveness update
        time_data = {0: 0.1, 1: 0.05, 2: -0.02} # Keys should be strings as per JSON standard
        time_data_str_keys = {str(k): v for k,v in time_data.items()}
        await params_manager.update_time_effectiveness(time_data_str_keys)
        fetched_time_effectiveness = params_manager.get_param('timeOfDayEffectiveness')
        print(f"Tijd-van-dag effectiviteit: {fetched_time_effectiveness}")
        assert fetched_time_effectiveness == time_data_str_keys

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

        reloaded_entry_rule_score = reloaded_manager.get_param('entryRulePatternScore', test_strategy_id)
        print(f"Hergeladen entryRulePatternScore voor {test_strategy_id}: {reloaded_entry_rule_score}")
        assert reloaded_entry_rule_score == 0.7

        reloaded_exit_rule_score = reloaded_manager.get_param('exitRulePatternScore', test_strategy_id)
        print(f"Hergeladen exitRulePatternScore voor {test_strategy_id}: {reloaded_exit_rule_score}")
        assert reloaded_exit_rule_score == 0.7

        print(f"Hergeladen ROI voor {test_strategy_id}: {reloaded_manager.get_param('minimal_roi', test_strategy_id)}")
        assert reloaded_manager.get_param("minimal_roi", test_strategy_id) == new_roi

        fetched_time_effectiveness_reloaded = reloaded_manager.get_param('timeOfDayEffectiveness')
        print(f"Hergeladen Tijd-van-dag effectiviteit: {fetched_time_effectiveness_reloaded}")
        assert fetched_time_effectiveness_reloaded == time_data_str_keys

        print("\nParamsManager tests succesvol afgerond.")

    asyncio.run(run_test_params_manager())

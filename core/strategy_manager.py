# core/strategy_manager.py
import json
import logging
import os
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import dotenv # Added for __main__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
STRATEGY_MEMORY_FILE = os.path.join(MEMORY_DIR, 'strategy_memory.json')

os.makedirs(MEMORY_DIR, exist_ok=True)

class StrategyManager:
    """
    Beheert strategieselectie, mutatie en geheugen.
    Vertaald van strategyManager.js en strategyMutator.js concepten.
    """

    def __init__(self):
        self.strategy_memory = self._load_strategy_memory()
        # BiasReflector needs to be an instance if get_best_strategy uses it.
        # This creates a dependency. For now, we'll assume it's set up externally
        # or mock it in tests if not passed in.
        self.bias_reflector: Optional[Any] = None # Placeholder for BiasReflector instance
        logger.info(f"StrategyManager geïnitialiseerd. Geheugen: {STRATEGY_MEMORY_FILE}")

    def _load_strategy_memory(self) -> Dict[str, Any]:
        try:
            if os.path.exists(STRATEGY_MEMORY_FILE) and os.path.getsize(STRATEGY_MEMORY_FILE) > 0:
                with open(STRATEGY_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {} # Return empty if file doesn't exist or is empty
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Strategie geheugenbestand {STRATEGY_MEMORY_FILE} niet gevonden of corrupt. Start met leeg geheugen.")
            return {}

    def _save_strategy_memory(self):
        try:
            with open(STRATEGY_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.strategy_memory, f, indent=2)
        except IOError as e:
            logger.error(f"Fout bij opslaan strategie geheugen: {e}")

    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Haalt de laatst bekende prestatie van een strategie op.
        """
        return self.strategy_memory.get(strategy_id, {}).get('performance', {"winRate": 0.0, "avgProfit": 0.0, "tradeCount": 0})

    async def update_strategy_performance(self, strategy_id: str, new_performance: Dict[str, Any]):
        """
        Werkt de prestatie van een strategie bij.
        """
        if strategy_id not in self.strategy_memory:
            self.strategy_memory[strategy_id] = {"parameters": {}, "performance": {}} # Ensure structure exists

        # Merge new performance with existing, new values overwrite old
        current_performance = self.strategy_memory[strategy_id].get('performance', {})
        current_performance.update(new_performance)
        current_performance["last_updated"] = datetime.now().isoformat()

        self.strategy_memory[strategy_id]['performance'] = current_performance

        await asyncio.to_thread(self._save_strategy_memory)
        logger.info(f"Prestatie voor strategie {strategy_id} bijgewerkt: {new_performance}")

    async def mutate_strategy(self, strategy_id: str, proposal: Dict[str, Any]):
        """
        Muteert een strategie op basis van een mutatievoorstel van de AI.
        Dit zou in de praktijk de Freqtrade strategie code zelf aanpassen of nieuwe parameters laden.
        """
        if not proposal or proposal.get('strategyId') != strategy_id:
            logger.warning(f"Ongeldig mutatievoorstel voor strategie {strategy_id}.")
            return False

        logger.info(f"Muteer strategie {strategy_id} met voorstel: {proposal.get('adjustments')}")

        if strategy_id not in self.strategy_memory:
            self.strategy_memory[strategy_id] = {"parameters": {}, "performance": {}}

        current_strategy_info = self.strategy_memory[strategy_id]
        # Ensure 'parameters' key exists
        current_params = current_strategy_info.get("parameters", {})

        parameter_changes = proposal.get('adjustments', {}).get('parameterChanges', {})
        if parameter_changes: # Only update if there are changes
            for param, value in parameter_changes.items():
                current_params[param] = value
                logger.debug(f"Parameter '{param}' van strategie {strategy_id} aangepast naar {value}.")
            current_strategy_info['parameters'] = current_params
            current_strategy_info['last_mutated'] = datetime.now().isoformat()
            current_strategy_info['mutation_rationale'] = proposal.get('rationale')
            current_strategy_info['last_mutation_proposal'] = proposal # Store the full proposal

            self.strategy_memory[strategy_id] = current_strategy_info
            await asyncio.to_thread(self._save_strategy_memory)
            logger.info(f"Strategie {strategy_id} succesvol gemuteerd met parameters: {current_params}.")
            return True
        else:
            logger.info(f"Geen parameter wijzigingen in voorstel voor strategie {strategy_id}. Mutatie niet uitgevoerd.")
            return False


    async def get_best_strategy(self, token: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        Selecteert de best presterende strategie voor een gegeven token en interval.
        Vertaald van getBestStrategy [DUO].
        Dit zou meerdere strategieën in het geheugen moeten vergelijken.
        Voor nu, retourneert het een mock of de "DUOAI_Strategy" met basisinfo.
        """
        # Als er meerdere strategieën zijn, kies de beste op basis van performance en bias
        # For this example, we assume "DUOAI_Strategy" is the one we manage.
        # In a multi-strategy system, you'd iterate through self.strategy_memory.keys()

        strategy_id = "DUOAI_Strategy" # De naam van je Freqtrade strategie

        if strategy_id not in self.strategy_memory:
            logger.warning(f"Strategie {strategy_id} niet gevonden in geheugen voor get_best_strategy.")
            # Initialize with default if not found, so it can be used
            self.strategy_memory[strategy_id] = {
                "parameters": {"emaPeriod": 20, "rsiThreshold": 70}, # Default parameters
                "performance": {"winRate": 0.0, "avgProfit": 0.0, "tradeCount": 0}
            }


        performance = self.get_strategy_performance(strategy_id)

        bias = 0.5 # Default bias
        if self.bias_reflector: # Check if bias_reflector instance is available
            bias = self.bias_reflector.get_bias_score(token, strategy_id)
        else:
            logger.warning("BiasReflector niet beschikbaar in StrategyManager. Gebruik default bias 0.5.")


        return {
            "id": strategy_id,
            "performance": performance,
            "bias": bias, # Learned bias for this token/strategy combo
            "parameters": self.strategy_memory.get(strategy_id, {}).get("parameters", {})
        }

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    # Corrected path for .env when running this script directly
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    dotenv.load_dotenv(dotenv_path)

    # Setup basic logging for the test
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])


    async def run_test_strategy_manager():
        strategy_manager = StrategyManager()
        test_strategy_id = "DUOAI_Strategy"
        test_token = "ETH/USDT"

        # Voor de test, initieer de bias_reflector die de strategy_manager aanroept
        # This shows the dependency. In a real app, BiasReflector might be passed via __init__
        # or accessed via a global/singleton pattern if appropriate.
        from core.bias_reflector import BiasReflector # Import here for test setup
        strategy_manager.bias_reflector = BiasReflector()


        print("\n--- Test StrategyManager ---")

        # Update prestatie
        print(f"Update prestatie voor {test_strategy_id}...")
        await strategy_manager.update_strategy_performance(test_strategy_id, {"winRate": 0.6, "avgProfit": 0.03, "tradeCount": 50})
        current_perf = strategy_manager.get_strategy_performance(test_strategy_id)
        print(f"Huidige prestatie: {current_perf}")
        assert current_perf['winRate'] == 0.6

        # Genereer een mock mutatievoorstel
        mock_proposal = {
            "strategyId": test_strategy_id,
            "adjustments": {
                "action": "strengthen",
                "parameterChanges": {"emaPeriod": 22, "rsiThreshold": 75}
            },
            "confidence": 0.85,
            "rationale": {"reason": "Goede prestaties, verhoog risico/vertrouwen."}
        }

        # Muteer strategie
        print(f"\nMuteer strategie {test_strategy_id}...")
        mutation_successful = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal)
        print(f"Mutatie succesvol: {mutation_successful}")
        mutated_params = strategy_manager.strategy_memory.get(test_strategy_id,{}).get('parameters')
        print(f"Nieuwe parameters na mutatie: {mutated_params}")
        assert mutated_params and mutated_params.get('emaPeriod') == 22

        # Get best strategy
        print(f"\nHaal beste strategie op voor {test_token} (interval 5m)...")
        best_strat_info = await strategy_manager.get_best_strategy(test_token, "5m")
        print(f"Beste strategie info: {best_strat_info}")
        assert best_strat_info is not None
        assert best_strat_info['id'] == test_strategy_id
        assert 'bias' in best_strat_info # Check if bias was retrieved

    asyncio.run(run_test_strategy_manager())

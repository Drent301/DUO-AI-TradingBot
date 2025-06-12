# tests/conftest.py
import pytest
import os
import json
import asyncio
from unittest.mock import patch, AsyncMock

# Configure logging for tests
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Adjust PYTHONPATH to import modules from core
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Project root


# --- Fixtures voor gedeelde testomgeving ---

@pytest.fixture(scope="function")
def event_loop():
    """Creëert een nieuw event loop voor elke testfunctie."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def run_around_tests():
    """Zorgt ervoor dat .env wordt geladen en memory bestanden worden opgeschoond voor elke test."""
    from dotenv import load_dotenv
    # Load .env file from project root
    load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))

    # Define paths to memory files
    memory_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../memory'))
    params_file = os.path.join(memory_dir, 'learned_params.json')
    cooldown_file = os.path.join(memory_dir, 'cooldown_memory.json')
    reflectie_log_file = os.path.join(memory_dir, 'reflectie-logboek.json')
    strategy_memory_file = os.path.join(memory_dir, 'strategy_memory.json')
    pretrain_log_file = os.path.join(memory_dir, 'pre_train_log.json')
    time_effectiveness_file = os.path.join(memory_dir, 'time_effectiveness.json')
    optimizer_log_file = os.path.join(memory_dir, 'ai_optimizer_log.json')
    cnn_model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/models')), 'cnn_patterns_model.pth')

    # Ensure memory directory exists
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/models')), exist_ok=True)


    # Clean up memory files before each test
    for f in [params_file, cooldown_file, reflectie_log_file, strategy_memory_file, pretrain_log_file, time_effectiveness_file, optimizer_log_file, cnn_model_path]:
        if os.path.exists(f):
            if os.path.isfile(f): # Delete file
                os.remove(f)
            elif os.path.isdir(f): # If it's a directory (like models dir), clear contents if needed
                pass # For now, don't delete models dir itself, just the .pth file within it

    # Ensure JSON files are created as empty lists/dicts if they are used by modules for writing
    for f in [params_file, cooldown_file, reflectie_log_file, strategy_memory_file, pretrain_log_file, time_effectiveness_file, optimizer_log_file]:
        if "json" in f and not os.path.exists(f):
            with open(f, 'w') as fh:
                json.dump([] if "log" in f or "reflectie" in f else {}, fh) # Empty list for logs, empty dict for others

    yield # Run the test

    # Clean up after tests (optional, can be commented out for debugging)
    # for f in [params_file, cooldown_file, reflectie_log_file, strategy_memory_file, pretrain_log_file, time_effectiveness_file, optimizer_log_file, cnn_model_path]:
    #     if os.path.exists(f):
    #         if os.path.isfile(f):
    #             os.remove(f)

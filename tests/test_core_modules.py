# tests/test_core_modules.py
import pytest
import os
import json
import asyncio
from unittest.mock import patch, mock_open, AsyncMock, MagicMock
from datetime import datetime, timedelta

# Ensure core modules can be imported
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Project root

# Modules to test
from params_manager import ParamsManager, PARAMS_FILE as DEFAULT_PARAMS_FILE
from cooldown_tracker import CooldownTracker, COOLDOWN_MEMORY_FILE as DEFAULT_COOLDOWN_FILE

# Use pytest-asyncio for async tests by marking the module
pytestmark = pytest.mark.asyncio

class TestParamsManager:
    async def test_initialization_no_file_uses_defaults(self, event_loop):
        """Test ParamsManager initialization when the params file does not exist uses defaults."""
        with patch('core.params_manager.os.path.exists', return_value=False), \
             patch('core.params_manager.PARAMS_FILE', "dummy/path/non_existent_params.json"): # Ensure it tries to load from a non-existent, mocked path
            manager = ParamsManager()
            # _load_params is called in __init__. If file doesn't exist, _get_default_params is called.
            assert manager._params == manager._get_default_params()
            assert "global" in manager._params # Check structure

    async def test_initialization_with_existing_file(self, event_loop):
        """Test ParamsManager initialization with an existing, valid JSON file."""
        expected_params = {"global": {"test_param": "test_value"}, "strategies": {}}
        # Use a mocked PARAMS_FILE path for this test
        mocked_params_file_path = "dummy/path/existent_params.json"

        with patch('core.params_manager.PARAMS_FILE', mocked_params_file_path), \
             patch('core.params_manager.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(expected_params))) as mock_file:

            manager = ParamsManager()
            assert manager._params == expected_params
            mock_file.assert_called_with(mocked_params_file_path, 'r', encoding='utf-8')

    async def test_initialization_with_empty_file_uses_defaults(self, event_loop):
        """Test ParamsManager initialization with an existing but empty JSON file uses defaults."""
        mocked_params_file_path = "dummy/path/empty_params.json"
        with patch('core.params_manager.PARAMS_FILE', mocked_params_file_path), \
             patch('core.params_manager.os.path.exists', return_value=True), \
             patch('core.params_manager.os.path.getsize', return_value=0), \
             patch('builtins.open', mock_open(read_data="")) as mock_file: # Mock open for the empty read

            manager = ParamsManager()
            assert manager._params == manager._get_default_params()
            # open would be called, but json.load would fail or be skipped due to empty content/size 0
            # The exact call depends on implementation, but it should reach _get_default_params

    async def test_initialization_with_corrupt_file_uses_defaults(self, event_loop):
        """Test ParamsManager initialization with a corrupt JSON file uses defaults."""
        mocked_params_file_path = "dummy/path/corrupt_params.json"
        with patch('core.params_manager.PARAMS_FILE', mocked_params_file_path), \
             patch('core.params_manager.os.path.exists', return_value=True), \
             patch('core.params_manager.os.path.getsize', return_value=100), \
             patch('builtins.open', mock_open(read_data='{"corrupt_json":}')) as mock_file:

            manager = ParamsManager()
            assert manager._params == manager._get_default_params()
            mock_file.assert_called_with(mocked_params_file_path, 'r', encoding='utf-8')


    @patch('core.params_manager.asyncio.to_thread') # Mocks the asyncio.to_thread call
    async def test_save_params_async(self, mock_to_thread, event_loop):
        """Test saving parameters calls asyncio.to_thread with json.dump."""
        manager = ParamsManager()
        test_params = {"test_key": "test_value"}
        manager._params = test_params # Directly set params for testing save

        # We need to ensure that the lambda passed to to_thread correctly calls json.dump
        # For this, we can inspect the first argument to mock_to_thread if it's a callable (lambda)
        # and then call it, or trust that asyncio.to_thread works and json.dump is called.
        # A more direct way is to mock the 'open' and 'json.dump' within the lambda's scope if possible,
        # but that's harder with 'to_thread'. Let's assume 'to_thread' itself is not what we're testing.

        # Mock 'open' and 'json.dump' that would be called inside the thread
        with patch('builtins.open', new_callable=mock_open) as mock_file_open, \
             patch('json.dump') as mock_json_dump:

            # Define a side effect for mock_to_thread that executes the passed function
            # This simulates the behavior of asyncio.to_thread for testing purposes
            async def side_effect_for_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            mock_to_thread.side_effect = side_effect_for_to_thread

            await manager._save_params()

            # Verify that open was called correctly by the lambda within to_thread
            mock_file_open.assert_called_with(DEFAULT_PARAMS_FILE, 'w', encoding='utf-8')
            # Verify that json.dump was called correctly
            mock_json_dump.assert_called_with(test_params, mock_file_open(), indent=2)
            mock_to_thread.assert_called_once() # Ensure to_thread itself was called

    @patch.object(ParamsManager, '_save_params', new_callable=AsyncMock) # Mock the async _save_params
    async def test_set_param_global(self, mock_save_params_method, event_loop):
        """Test setting a global parameter."""
        manager = ParamsManager()
        manager._params = manager._get_default_params() # Start with defaults for predictability

        await manager.set_param("maxTradeRiskPct", 0.05)
        assert manager._params["global"]["maxTradeRiskPct"] == 0.05
        mock_save_params_method.assert_called_once()

    @patch.object(ParamsManager, '_save_params', new_callable=AsyncMock)
    async def test_set_param_strategy_specific(self, mock_save_params_method, event_loop):
        """Test setting a strategy-specific parameter."""
        manager = ParamsManager()
        manager._params = manager._get_default_params()
        strategy_id = "DUOAI_Strategy"

        await manager.set_param("entryConvictionThreshold", 0.85, strategy_id=strategy_id)
        assert manager._params["strategies"][strategy_id]["entryConvictionThreshold"] == 0.85
        mock_save_params_method.assert_called_once()

    @patch.object(ParamsManager, '_save_params', new_callable=AsyncMock)
    async def test_set_param_new_strategy(self, mock_save_params_method, event_loop):
        """Test setting a parameter for a new strategy."""
        manager = ParamsManager()
        manager._params = manager._get_default_params()
        new_strategy_id = "NewStrategy_Test"

        await manager.set_param("someNewParam", "newValue", strategy_id=new_strategy_id)
        assert manager._params["strategies"][new_strategy_id]["someNewParam"] == "newValue"
        mock_save_params_method.assert_called_once()

    async def test_get_param_retrieval_logic(self, event_loop):
        """Test get_param retrieval: strategy-specific, global, default."""
        manager = ParamsManager()
        default_params = manager._get_default_params()

        # Setup specific state for testing get_param
        manager._params = {
            "global": {"maxTradeRiskPct": 0.025, "cooldownDurationSeconds": 600},
            "strategies": {
                "DUOAI_Strategy": {"entryConvictionThreshold": 0.75, "stoploss": -0.15},
                "OTHER_Strategy": {"exitConvictionDropTrigger": 0.3}
            },
            "timeOfDayEffectiveness": {"10": 0.5} # Example root-level param
        }

        # 1. Get strategy-specific param
        assert manager.get_param("entryConvictionThreshold", strategy_id="DUOAI_Strategy") == 0.75
        # 2. Get global param when strategy-specific does not exist for that key
        assert manager.get_param("cooldownDurationSeconds", strategy_id="DUOAI_Strategy") == 600
        # 3. Get global param directly
        assert manager.get_param("maxTradeRiskPct") == 0.025
        # 4. Get root-level param
        assert manager.get_param("timeOfDayEffectiveness") == {"10": 0.5}
        # 5. Fallback to default global param if not in current _params
        assert manager.get_param("slippageTolerancePct") == default_params["global"]["slippageTolerancePct"]
        # 6. Fallback to default strategy param if not in current _params for that strategy
        assert manager.get_param("cnnPatternWeight", strategy_id="DUOAI_Strategy") == default_params["strategies"]["DUOAI_Strategy"]["cnnPatternWeight"]
        # 7. Param not found anywhere
        assert manager.get_param("non_existent_param_total") is None
        assert manager.get_param("non_existent_param_strat", strategy_id="DUOAI_Strategy") is None
        # 8. Fallback for a key that is strategy-specific by default, but trying to get for a strategy not in current custom _params
        assert manager.get_param("entryConvictionThreshold", strategy_id="STRAT_NOT_IN_CUSTOM") == default_params["strategies"]["DUOAI_Strategy"]["entryConvictionThreshold"] # Assumes DUOAI_Strategy is a known default structure

class TestCooldownTracker:
    async def test_initialization_no_file_uses_empty(self, event_loop):
        """Test CooldownTracker initialization when the cooldown file does not exist uses empty state."""
        with patch('core.cooldown_tracker.os.path.exists', return_value=False), \
             patch('core.cooldown_tracker.COOLDOWN_MEMORY_FILE', "dummy/path/non_existent_cooldowns.json"):
            tracker = CooldownTracker()
            assert tracker._cooldown_state == {}

    async def test_initialization_with_existing_file(self, event_loop):
        """Test CooldownTracker initialization with an existing, valid JSON file."""
        expected_state = {"token1/stratA": {"end_time": "2023-01-01T12:00:00Z", "reason": "test"}}
        mocked_cooldown_file_path = "dummy/path/existent_cooldowns.json"

        with patch('core.cooldown_tracker.COOLDOWN_MEMORY_FILE', mocked_cooldown_file_path), \
             patch('core.cooldown_tracker.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(expected_state))) as mock_file:

            tracker = CooldownTracker()
            assert tracker._cooldown_state == expected_state
            mock_file.assert_called_with(mocked_cooldown_file_path, 'r', encoding='utf-8')

    @patch('core.cooldown_tracker.asyncio.to_thread')
    async def test_save_cooldown_state_async(self, mock_to_thread, event_loop):
        """Test saving cooldown state calls asyncio.to_thread with json.dump."""
        tracker = CooldownTracker()
        test_state = {"token1/stratA": {"end_time": "2023-01-01T13:00:00Z"}}
        tracker._cooldown_state = test_state

        with patch('builtins.open', new_callable=mock_open) as mock_file_open, \
             patch('json.dump') as mock_json_dump:

            async def side_effect_for_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            mock_to_thread.side_effect = side_effect_for_to_thread

            await tracker._save_cooldown_state()

            mock_file_open.assert_called_with(DEFAULT_COOLDOWN_FILE, 'w', encoding='utf-8')
            mock_json_dump.assert_called_with(test_state, mock_file_open(), indent=2)
            mock_to_thread.assert_called_once()

    @patch.object(CooldownTracker, '_save_cooldown_state', new_callable=AsyncMock)
    @patch('core.cooldown_tracker.datetime') # Mock datetime globally for this test
    async def test_activate_cooldown(self, mock_datetime_module, mock_save_method, event_loop):
        """Test activating a cooldown."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime_module.now.return_value = mock_now

        tracker = CooldownTracker()
        # Mock ParamsManager instance within CooldownTracker
        tracker.params_manager = AsyncMock(spec=ParamsManager)
        tracker.params_manager.get_param = MagicMock(return_value=300) # 300s cooldown

        token = "ETH/USDT"
        strategy_id = "TestStrategy"
        reason = "test_activation"

        await tracker.activate_cooldown(token, strategy_id, reason)

        expected_end_time = mock_now + timedelta(seconds=300)
        assert token in tracker._cooldown_state
        assert strategy_id in tracker._cooldown_state[token]
        assert tracker._cooldown_state[token][strategy_id]["end_time"] == expected_end_time.isoformat()
        assert tracker._cooldown_state[token][strategy_id]["reason"] == reason
        assert tracker._cooldown_state[token][strategy_id]["activated_at"] == mock_now.isoformat()

        mock_save_method.assert_called_once()
        tracker.params_manager.get_param.assert_called_with("cooldownDurationSeconds", strategy_id=None)

    @patch('core.cooldown_tracker.datetime')
    async def test_is_cooldown_active_true(self, mock_datetime_module, event_loop):
        """Test is_cooldown_active returns True when cooldown is active."""
        tracker = CooldownTracker()
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime_module.now.return_value = mock_now

        token = "BTC/USDT"
        strategy_id = "MainStrat"
        active_end_time = mock_now + timedelta(seconds=100)
        tracker._cooldown_state = {token: {strategy_id: {"end_time": active_end_time.isoformat()}}}

        assert tracker.is_cooldown_active(token, strategy_id) is True

    @patch('core.cooldown_tracker.datetime')
    async def test_is_cooldown_active_false_expired(self, mock_datetime_module, event_loop):
        """Test is_cooldown_active returns False when cooldown has expired."""
        tracker = CooldownTracker()
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime_module.now.return_value = mock_now

        token = "ADA/USDT"
        strategy_id = "AltStrat"
        expired_end_time = mock_now - timedelta(seconds=100)
        tracker._cooldown_state = {token: {strategy_id: {"end_time": expired_end_time.isoformat()}}}

        assert tracker.is_cooldown_active(token, strategy_id) is False
        # Optional: check if expired entry is cleaned up (if implemented, currently not)
        # assert token not in tracker._cooldown_state

    async def test_is_cooldown_active_item_not_present(self, event_loop):
        """Test is_cooldown_active returns False for an item not in cooldowns."""
        tracker = CooldownTracker()
        tracker._cooldown_state = {} # Ensure empty state
        assert tracker.is_cooldown_active("NONEXISTENT/TOKEN", "ANYSTRAT") is False

    @patch.object(CooldownTracker, '_save_cooldown_state', new_callable=AsyncMock)
    async def test_deactivate_cooldown(self, mock_save_method, event_loop):
        """Test deactivating an active cooldown."""
        tracker = CooldownTracker()
        token = "LINK/USDT"
        strategy_id = "SwingStrat"
        # Setup an active cooldown
        tracker._cooldown_state = {
            token: {strategy_id: {"end_time": (datetime.now() + timedelta(hours=1)).isoformat()}}
        }

        await tracker.deactivate_cooldown(token, strategy_id)

        assert token not in tracker._cooldown_state or strategy_id not in tracker._cooldown_state[token]
        mock_save_method.assert_called_once()

    @patch.object(CooldownTracker, '_save_cooldown_state', new_callable=AsyncMock)
    async def test_deactivate_cooldown_nonexistent(self, mock_save_method, event_loop):
        """Test deactivating a non-existent cooldown does not error and does not save."""
        tracker = CooldownTracker()
        tracker._cooldown_state = {}

        await tracker.deactivate_cooldown("XYZ/USDT", "StratX")

        assert "XYZ/USDT" not in tracker._cooldown_state
        mock_save_method.assert_not_called()

# Note: If ParamsManager or CooldownTracker had synchronous file I/O in __init__
# that needed to be truly async or controlled in tests, you might need more complex
# patching for their __init__ if they performed blocking I/O not handled by conftest.py's setup.
# However, the current structure of these classes seems to load data in a way that's
# manageable with patching os.path.exists, builtins.open, and os.path.getsize for __init__.
# Async file operations (_save_params, _save_cooldown_state) are correctly mocked with AsyncMock for their respective classes.
# The `event_loop` fixture provided by pytest-asyncio is implicitly used for async tests.
# No explicit `tmp_path` usage in CooldownTracker tests as it also uses a global default file path.
# Mocks for `params_manager` instance within `CooldownTracker` are handled per test method.
# `conftest.py` is assumed to handle the cleanup of DEFAULT_PARAMS_FILE and DEFAULT_COOLDOWN_FILE between test runs.
# If `conftest.py` doesn't create these files as empty JSONs initially, tests for empty/corrupt files might need adjustment.
# The current tests assume `conftest.py` removes files, so `os.path.exists` will be False unless explicitly mocked.
# For tests like `test_initialization_with_existing_file`, we are mocking the path constants
# to use a path that `builtins.open` can be reliably mocked for, without interfering with global state
# that might be managed by `conftest.py`.

# Final check on ParamsManager.get_param default logic when strategy_id is provided but key is global:
# It should fall back to global if strategy-specific is not found for that key.
# If key is not in global either, then it falls back to default params structure.
# If key is not in default params structure at all, it returns None. This seems covered.

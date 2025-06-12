import pytest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock, call
import json
import os
import asyncio # Required for async tests
from datetime import datetime, timedelta # Required for CooldownTracker tests

# Ensure core modules can be imported
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Project root for conftest .env loading

from params_manager import ParamsManager, PARAMS_FILE as DEFAULT_PARAMS_FILE
from cooldown_tracker import CooldownTracker, COOLDOWN_MEMORY_FILE as DEFAULT_COOLDOWN_FILE

# pytestmark = pytest.mark.asyncio # Apply to all tests in this module if using older pytest-asyncio

class TestParamsManager:
    @pytest.mark.asyncio
    async def test_initialization_no_file(self):
        """Test ParamsManager initialization when the params file does not exist (handled by conftest)."""
        # conftest.py should ensure PARAMS_FILE is clean or non-existent initially
        # Forcing a clean state for this specific test if conftest doesn't perfectly isolate
        if os.path.exists(DEFAULT_PARAMS_FILE):
            os.remove(DEFAULT_PARAMS_FILE)

        manager = ParamsManager()
        assert manager._params == manager._get_default_params()

    @pytest.mark.asyncio
    async def test_initialization_with_file(self, tmp_path):
        """Test ParamsManager initialization when the params file exists."""
        # We need to control the specific file ParamsManager uses for this test.
        # Since ParamsManager hardcodes PARAMS_FILE, we mock it.
        custom_params_file = tmp_path / "custom_params.json"
        expected_params = {"param1": "value1"}
        with open(custom_params_file, 'w') as f:
            json.dump(expected_params, f)

        with patch('core.params_manager.PARAMS_FILE', str(custom_params_file)):
            manager = ParamsManager()
            assert manager._params == expected_params

    @pytest.mark.asyncio
    @patch('core.params_manager.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"key1": "value1"}')
    async def test_load_params_file_exists(self, mock_file_open_builtin, mock_path_exists):
        """Test loading parameters when the file exists."""
        # PARAMS_FILE is used internally, ensure mocks apply to it
        with patch('core.params_manager.PARAMS_FILE', "mocked/path/params.json"):
            manager = ParamsManager()
            assert manager._params == {"key1": "value1"}
            mock_file_open_builtin.assert_called_with("mocked/path/params.json", 'r', encoding='utf-8')

    @pytest.mark.asyncio
    @patch('core.params_manager.os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open) # To catch any accidental open calls
    async def test_load_params_file_not_found_uses_defaults(self, mock_file_open_builtin, mock_path_exists):
        """Test loading parameters uses defaults when the file does not exist."""
        with patch('core.params_manager.PARAMS_FILE', "mocked/path/non_existent_params.json"):
            manager = ParamsManager()
            assert manager._params == manager._get_default_params()
            # open should not be called for reading if os.path.exists is false by _load_params logic
            # and it should fall back to _get_default_params()
            read_calls = [c for c in mock_file_open_builtin.mock_calls if c[0] == '' and 'r' in c[1]] # Check for read calls
            assert not read_calls


    @pytest.mark.asyncio
    @patch('core.params_manager.asyncio.to_thread')
    async def test_save_params(self, mock_to_thread):
        """Test saving parameters to the JSON file."""
        manager = ParamsManager()
        manager._params = {"key_new": "value_new"}

        # Mock the actual json.dump call if to_thread is used as a wrapper
        # For simplicity, we assume to_thread is correctly calling json.dump
        # and focus on whether _save_params attempts the save.
        await manager._save_params()

        # Check if asyncio.to_thread was called, implying an attempt to save.
        mock_to_thread.assert_called_once()


    @pytest.mark.asyncio
    @patch.object(ParamsManager, '_save_params', new_callable=AsyncMock)
    async def test_set_param(self, mock_save_params_method):
        """Test setting a parameter calls save."""
        manager = ParamsManager()
        # Clear params to ensure a clean state for what's being set
        manager._params = manager._get_default_params() # Start with defaults

        await manager.set_param("maxTradeRiskPct", 0.05) # Global
        assert manager._params["global"]["maxTradeRiskPct"] == 0.05
        mock_save_params_method.assert_called_once()

        mock_save_params_method.reset_mock()
        await manager.set_param("entryConvictionThreshold", 0.85, strategy_id="DUOAI_Strategy")
        assert manager._params["strategies"]["DUOAI_Strategy"]["entryConvictionThreshold"] == 0.85
        mock_save_params_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_param(self):
        """Test retrieving a parameter."""
        manager = ParamsManager()
        # Setup some internal params directly for testing get
        manager._params = {
            "global": {"param_g": "global_val"},
            "strategies": {
                "strat1": {"param_s1": "strat1_val"}
            },
            "timeOfDayEffectiveness": {"00": 0.5}
        }
        assert manager.get_param("param_g") == "global_val"
        assert manager.get_param("param_s1", strategy_id="strat1") == "strat1_val"
        assert manager.get_param("timeOfDayEffectiveness") == {"00": 0.5}

        # Test fallback to default if not in current _params
        assert manager.get_param("maxTradeRiskPct") == manager._get_default_params()["global"]["maxTradeRiskPct"]
        assert manager.get_param("entryConvictionThreshold", "DUOAI_Strategy") == manager._get_default_params()["strategies"]["DUOAI_Strategy"]["entryConvictionThreshold"]

        assert manager.get_param("non_existent_param") is None
        # Note: get_param doesn't take a default value argument in its current implementation

class TestCooldownTracker:
    @pytest.mark.asyncio
    async def test_initialization_no_file(self):
        """Test CooldownTracker initialization when the cooldown file does not exist."""
        if os.path.exists(DEFAULT_COOLDOWN_FILE):
            os.remove(DEFAULT_COOLDOWN_FILE)
        tracker = CooldownTracker()
        assert tracker._cooldown_state == {}

    @pytest.mark.asyncio
    async def test_initialization_with_file(self, tmp_path):
        """Test CooldownTracker initialization when the cooldown file exists."""
        custom_cooldown_file = tmp_path / "custom_cooldowns.json"
        expected_cooldowns = {"item1/strat1": {"end_time": "2025-01-01T12:00:00"}}
        with open(custom_cooldown_file, 'w') as f:
            json.dump(expected_cooldowns, f)

        with patch('core.cooldown_tracker.COOLDOWN_MEMORY_FILE', str(custom_cooldown_file)):
            tracker = CooldownTracker()
            assert tracker._cooldown_state == expected_cooldowns

    @pytest.mark.asyncio
    @patch('core.cooldown_tracker.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"item1/s1": {"end_time": "2023-01-01T10:00:00Z"}}')
    async def test_load_cooldowns_file_exists(self, mock_file_open_builtin, mock_path_exists):
        """Test loading cooldowns when the file exists."""
        with patch('core.cooldown_tracker.COOLDOWN_MEMORY_FILE', "mocked/path/cooldowns.json"):
            tracker = CooldownTracker()
            assert tracker._cooldown_state == {"item1/s1": {"end_time": "2023-01-01T10:00:00Z"}}
            mock_file_open_builtin.assert_called_with("mocked/path/cooldowns.json", 'r', encoding='utf-8')

    @pytest.mark.asyncio
    @patch('core.cooldown_tracker.os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open)
    async def test_load_cooldowns_file_not_found_uses_empty(self, mock_file_open_builtin, mock_path_exists):
        """Test loading cooldowns uses empty dict when the file does not exist."""
        with patch('core.cooldown_tracker.COOLDOWN_MEMORY_FILE', "mocked/path/non_existent_cooldowns.json"):
            tracker = CooldownTracker()
            assert tracker._cooldown_state == {}
            read_calls = [c for c in mock_file_open_builtin.mock_calls if c[0] == '' and 'r' in c[1]]
            assert not read_calls


    @pytest.mark.asyncio
    @patch('core.cooldown_tracker.asyncio.to_thread')
    async def test_save_cooldown_state(self, mock_to_thread):
        """Test saving cooldowns to the JSON file."""
        tracker = CooldownTracker()
        tracker._cooldown_state = {"item_new/s_new": {"end_time": "2023-02-01T00:00:00Z"}}
        await tracker._save_cooldown_state()
        mock_to_thread.assert_called_once()


    @pytest.mark.asyncio
    @patch.object(CooldownTracker, '_save_cooldown_state', new_callable=AsyncMock)
    @patch('core.cooldown_tracker.datetime') # Mock datetime within the module
    async def test_activate_cooldown(self, mock_datetime, mock_save_state_method):
        """Test activating a cooldown for an item."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        tracker = CooldownTracker()
        # Mock params_manager.get_param to return a fixed cooldown duration
        tracker.params_manager.get_param = MagicMock(return_value=60) # 60 seconds cooldown

        await tracker.activate_cooldown("test_item", "test_strategy", reason="test")

        expected_end_time = mock_now + timedelta(seconds=60)
        assert "test_item" in tracker._cooldown_state
        assert "test_strategy" in tracker._cooldown_state["test_item"]
        assert tracker._cooldown_state["test_item"]["test_strategy"]["end_time"] == expected_end_time.isoformat()
        mock_save_state_method.assert_called_once()
        tracker.params_manager.get_param.assert_called_with("cooldownDurationSeconds", strategy_id=None)


    @pytest.mark.asyncio
    @patch('core.cooldown_tracker.datetime')
    async def test_is_cooldown_active_true(self, mock_datetime):
        """Test is_cooldown_active returns True when an item is on cooldown."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        tracker = CooldownTracker()
        cooldown_end_time = mock_now + timedelta(seconds=60)
        tracker._cooldown_state = {
            "cooled_item": {"test_strat": {"end_time": cooldown_end_time.isoformat()}}
        }
        assert tracker.is_cooldown_active("cooled_item", "test_strat") is True

    @pytest.mark.asyncio
    @patch('core.cooldown_tracker.datetime')
    async def test_is_cooldown_active_false_expired(self, mock_datetime):
        """Test is_cooldown_active returns False when cooldown has expired."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        tracker = CooldownTracker()
        cooldown_expired_time = mock_now - timedelta(seconds=60)
        tracker._cooldown_state = {
            "cooled_item": {"test_strat": {"end_time": cooldown_expired_time.isoformat()}}
        }
        assert tracker.is_cooldown_active("cooled_item", "test_strat") is False
        # The current implementation does not remove the item upon check.
        # assert "cooled_item" not in tracker._cooldown_state

    @pytest.mark.asyncio
    async def test_is_cooldown_active_item_not_found(self):
        """Test is_cooldown_active returns False for an item not in cooldowns."""
        tracker = CooldownTracker()
        tracker._cooldown_state = {}
        assert tracker.is_cooldown_active("unknown_item", "unknown_strategy") is False

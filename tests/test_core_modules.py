import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import json
import time
import os

# Ensure core modules can be imported
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Project root for conftest .env loading

from params_manager import ParamsManager
from cooldown_tracker import CooldownTracker

class TestParamsManager:
    def test_initialization_no_file(self, tmp_path):
        """Test ParamsManager initialization when the params file does not exist."""
        params_file = tmp_path / "params.json"
        manager = ParamsManager(str(params_file))
        assert manager.params_file == str(params_file)
        assert manager.params == {}

    def test_initialization_with_file(self, tmp_path):
        """Test ParamsManager initialization when the params file exists."""
        params_file = tmp_path / "params.json"
        expected_params = {"param1": "value1"}
        with open(params_file, 'w') as f:
            json.dump(expected_params, f)

        manager = ParamsManager(str(params_file))
        assert manager.params_file == str(params_file)
        assert manager.params == expected_params

    @patch('core.params_manager.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"key1": "value1"}')
    def test_load_params_file_exists(self, mock_file_open, mock_path_exists):
        """Test loading parameters when the file exists."""
        # The params_file path used here is nominal as open is mocked.
        # However, conftest.py creates 'memory/learned_params.json' which might be default
        # if no path is given to ParamsManager, ensure to use a specific path for clarity.
        manager = ParamsManager("dummy/path/params.json")
        # load_params is called in __init__, so params should be loaded
        assert manager.params == {"key1": "value1"}
        mock_file_open.assert_called_with("dummy/path/params.json", 'r')

    @patch('core.params_manager.os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_params_file_not_found(self, mock_file_open, mock_path_exists):
        """Test loading parameters when the file does not exist initially."""
        manager = ParamsManager("dummy/path/non_existent_params.json")
        assert manager.params == {} # Should initialize to empty dict
        # open should not be called for reading if os.path.exists is false
        mock_file_open.assert_not_called()


    @patch('builtins.open', new_callable=mock_open)
    @patch('core.params_manager.json.dump')
    def test_save_params(self, mock_json_dump, mock_file_open):
        """Test saving parameters to the JSON file."""
        manager = ParamsManager("dummy/path/params.json")
        manager.params = {"key_new": "value_new"}
        manager.save_params()
        mock_file_open.assert_called_with("dummy/path/params.json", 'w')
        mock_json_dump.assert_called_with({"key_new": "value_new"}, mock_file_open(), indent=4)

    @patch.object(ParamsManager, 'save_params') # Mock save_params to avoid file I/O
    def test_update_param(self, mock_save_params):
        """Test updating a parameter."""
        manager = ParamsManager("dummy/path/params.json")
        manager.params = {"existing_param": "old_value"} # Initialize with some params
        manager.update_param("existing_param", "new_value")
        assert manager.params["existing_param"] == "new_value"
        manager.update_param("new_param", "another_value")
        assert manager.params["new_param"] == "another_value"
        assert mock_save_params.call_count == 2 # save_params should be called after each update

    def test_get_param(self):
        """Test retrieving a parameter."""
        manager = ParamsManager("dummy/path/params.json")
        manager.params = {"param1": "value1", "param2": 100}
        assert manager.get_param("param1") == "value1"
        assert manager.get_param("param2") == 100
        assert manager.get_param("non_existent_param") is None # Default for non-existent
        assert manager.get_param("non_existent_param_with_default", "default_val") == "default_val"

# Placeholder for future tests if needed
# def test_placeholder():
#     assert True

class TestCooldownTracker:
    def test_initialization_no_file(self, tmp_path):
        """Test CooldownTracker initialization when the cooldown file does not exist."""
        cooldown_file = tmp_path / "cooldowns.json"
        tracker = CooldownTracker(str(cooldown_file))
        assert tracker.cooldown_file == str(cooldown_file)
        assert tracker.cooldowns == {}

    def test_initialization_with_file(self, tmp_path):
        """Test CooldownTracker initialization when the cooldown file exists."""
        cooldown_file = tmp_path / "cooldowns.json"
        expected_cooldowns = {"item1": time.time() + 100}
        with open(cooldown_file, 'w') as f:
            json.dump(expected_cooldowns, f)

        tracker = CooldownTracker(str(cooldown_file))
        assert tracker.cooldown_file == str(cooldown_file)
        assert tracker.cooldowns == expected_cooldowns

    @patch('core.cooldown_tracker.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='{"item1": 1234567890.0}')
    def test_load_cooldowns_file_exists(self, mock_file_open, mock_path_exists):
        """Test loading cooldowns when the file exists."""
        tracker = CooldownTracker("dummy/path/cooldowns.json")
        assert tracker.cooldowns == {"item1": 1234567890.0}
        mock_file_open.assert_called_with("dummy/path/cooldowns.json", 'r')

    @patch('core.cooldown_tracker.os.path.exists', return_value=False)
    @patch('builtins.open', new_callable=mock_open)
    def test_load_cooldowns_file_not_found(self, mock_file_open, mock_path_exists):
        """Test loading cooldowns when the file does not exist."""
        tracker = CooldownTracker("dummy/path/non_existent_cooldowns.json")
        assert tracker.cooldowns == {}
        mock_file_open.assert_not_called()

    @patch('builtins.open', new_callable=mock_open)
    @patch('core.cooldown_tracker.json.dump')
    def test_save_cooldowns(self, mock_json_dump, mock_file_open):
        """Test saving cooldowns to the JSON file."""
        tracker = CooldownTracker("dummy/path/cooldowns.json")
        tracker.cooldowns = {"item_new": 9876543210.0}
        tracker._save_cooldowns() # Note: _save_cooldowns is the method name in the class
        mock_file_open.assert_called_with("dummy/path/cooldowns.json", 'w')
        mock_json_dump.assert_called_with({"item_new": 9876543210.0}, mock_file_open(), indent=4)

    @patch('core.cooldown_tracker.time.time')
    @patch.object(CooldownTracker, '_save_cooldowns')
    def test_set_cooldown(self, mock_save_cooldowns, mock_time):
        """Test setting a cooldown for an item."""
        mock_current_time = 1000.0
        mock_time.return_value = mock_current_time

        tracker = CooldownTracker("dummy/path/cooldowns.json")
        tracker.set_cooldown("test_item", 60) # 60 seconds cooldown

        assert "test_item" in tracker.cooldowns
        assert tracker.cooldowns["test_item"] == mock_current_time + 60
        mock_save_cooldowns.assert_called_once()

    @patch('core.cooldown_tracker.time.time')
    def test_is_on_cooldown_true(self, mock_time):
        """Test is_on_cooldown returns True when an item is on cooldown."""
        tracker = CooldownTracker("dummy/path/cooldowns.json")
        # Set up cooldowns directly for testing this method without relying on set_cooldown's mocks
        future_time = 1000.0
        tracker.cooldowns = {"cooled_item": future_time + 60} # Cooldown expires at 1060.0

        mock_time.return_value = future_time # Current time is 1000.0
        assert tracker.is_on_cooldown("cooled_item") is True

    @patch('core.cooldown_tracker.time.time')
    def test_is_on_cooldown_false_expired(self, mock_time):
        """Test is_on_cooldown returns False when cooldown has expired."""
        tracker = CooldownTracker("dummy/path/cooldowns.json")
        past_time = 1000.0
        tracker.cooldowns = {"cooled_item": past_time - 60} # Cooldown expired at 940.0

        mock_time.return_value = past_time # Current time is 1000.0
        assert tracker.is_on_cooldown("cooled_item") is False
        # Also check that the item is removed from cooldowns after check if expired
        assert "cooled_item" not in tracker.cooldowns

    def test_is_on_cooldown_item_not_found(self):
        """Test is_on_cooldown returns False for an item not in cooldowns."""
        tracker = CooldownTracker("dummy/path/cooldowns.json")
        tracker.cooldowns = {} # Ensure no items are in cooldown
        assert tracker.is_on_cooldown("unknown_item") is False

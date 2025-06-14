import pytest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock, call
import json
import os
import asyncio # Required for async tests
from datetime import datetime, timedelta # Required for CooldownTracker tests

# Ensure core modules can be imported
import sys
import torch # Added for mocking torch.save in PreTrainer tests
import numpy as np # Added for np.array used in test_cnn_patterns_model_scaler_loading
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

# Imports for PreTrainer tests
import pandas as pd
from core.pre_trainer import PreTrainer, CnnPatternsForLabeling # Assuming this is the correct import path
# from core.cnn_patterns import CNNPatterns # If CNNPatterns is separate and needs direct mocking

# Constants for PreTrainer tests (adjust paths as necessary)
MODELS_DIR = "models/pre_trained"
PRETRAIN_LOG_FILE = "logs/pretrain_activity.log"

class TestPreTrainer:
    @pytest.mark.asyncio
    async def test_pre_trainer_prepare_training_data_labeling(self):
        # Mock ParamsManager and its get_param method
        mock_params_manager = MagicMock(spec=ParamsManager)
        mock_params_manager.get_param.side_effect = self._get_param_side_effect

        # Instantiate PreTrainer with the mocked ParamsManager
        # CooldownTracker might not be strictly necessary for this specific method,
        # but pass a mock if PreTrainer constructor requires it.
        mock_cooldown_tracker = MagicMock(spec=CooldownTracker)
        pre_trainer = PreTrainer(params_manager=mock_params_manager, cooldown_tracker=mock_cooldown_tracker)

        # Create dummy DataFrame
        data = {
            'open':   [100, 102, 101, 103, 105, 104, 106, 107, 108, 110, 109, 112, 110, 111, 113, 115, 114, 116],
            'high':   [101, 103, 102, 104, 106, 105, 107, 108, 109, 111, 110, 113, 111, 112, 114, 116, 115, 117],
            'low':    [99,  101, 100, 102, 104, 103, 105, 106, 107, 109, 108, 111, 109, 110, 112, 114, 113, 115],
            'close':  [101, 101, 102, 104, 104, 105, 106, 107, 108, 110, 109, 112, 110, 111, 113, 115, 114, 116], # len 18
            'volume': [1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]
        }
        # Need enough data points to account for future_lookahead_bull/bear and sequence_length
        # Example: future_lookahead_bull = 5, sequence_length = 10. Min length = 15 for one usable sequence.
        # Let's make it 18 to be safe for a few operations.
        index = pd.to_datetime([datetime(2023, 1, 1, i, 0, 0) for i in range(len(data['close']))])
        dummy_df = pd.DataFrame(data, index=index)

        # --- BullFlag Scenario ---
        # For a bullFlag at index `i`, we need:
        # (dummy_df['close'][i + future_lookahead_bull] - dummy_df['close'][i]) / dummy_df['close'][i] > price_increase_threshold_bull
        # Let future_lookahead_bull = 3, price_increase_threshold_bull = 0.02
        # Candle 0: close=101. Future close (candle 3) = 104. (104-101)/101 = 0.0297 > 0.02. Label should be 1.
        # Candle 1: close=101. Future close (candle 4) = 104. (104-101)/101 = 0.0297 > 0.02. Label should be 1.
        # Candle 2: close=102. Future close (candle 5) = 105. (105-102)/102 = 0.0294 > 0.02. Label should be 1.

        # --- BearishEngulfing Scenario ---
        # For a bearishEngulfing at index `i`, we need CNNPatterns._detect_engulfing to return "bearishEngulfing"
        # and then (dummy_df['close'][i] - dummy_df['close'][i + future_lookahead_bear]) / dummy_df['close'][i] > price_decrease_threshold_bear
        # Let future_lookahead_bear = 2, price_decrease_threshold_bear = 0.01
        # Let's assume _detect_engulfing finds a bearish engulfing pattern at index 10 (close=109).
        # We need (109 - close[12]=110) / 109 = -0.009 (not > 0.01). Label 0.
        # Let's adjust data for a bearish engulfing label 1.
        # Suppose candle 10 (close=109) is bearishEngulfing.
        # We need (close[10] - close[10+2]) / close[10] > price_decrease_threshold_bear
        # (109 - close[12]=110) / 109 = -0.009. This will be label 0.
        # Let's make candle 12 close lower: data['close'][12] = 105
        # Then (109 - 105) / 109 = 4/109 = 0.036 > 0.01. Label should be 1.
        # We also need _detect_engulfing to return "bearishEngulfing" for candle 10.
        # A bearish engulfing: previous candle (9) is green, current candle (10) is red and engulfs body of (9).
        # Candle 9: open=110, close=109 (Red, if we consider open>close, but usually data is open<close for green)
        # Let's fix candle 9: open=108, close=110 (Green)
        # Candle 10: open=111, close=109 (Red, engulfs 108-110 body? No, open is higher)
        # Needs open[10] < close[9] and close[10] < open[9]
        # Let candle 9: open=108, close=110 (Green body: 108-110)
        # Let candle 10: open=110.5, close=107.5 (Red body: 107.5-110.5, engulfs 108-110)
        # This means data['open'][9]=108, data['close'][9]=110
        # data['open'][10]=110.5, data['close'][10]=107.5
        # And for label 1: (close[10] - close[12]) / close[10] > price_decrease_threshold_bear
        # (107.5 - close[12]) / 107.5 > 0.01.
        # Let data['close'][12] be 105. (107.5 - 105) / 107.5 = 2.5 / 107.5 = 0.023 > 0.01. This should be label 1.

        # Update data for bearish engulfing
        dummy_df.loc[dummy_df.index[9], 'open'] = 108
        dummy_df.loc[dummy_df.index[9], 'close'] = 110
        dummy_df.loc[dummy_df.index[10], 'open'] = 110.5
        dummy_df.loc[dummy_df.index[10], 'close'] = 107.5
        dummy_df.loc[dummy_df.index[12], 'close'] = 105 # For label 1

        # Mock CnnPatternsForLabeling used by PreTrainer
        # The instance is created inside prepare_training_data, so we patch the class.
        with patch('core.pre_trainer.CnnPatternsForLabeling') as MockCnnPatterns:
            mock_cnn_instance = MockCnnPatterns.return_value

            # Simulate _detect_engulfing behavior
            def mock_detect_engulfing_side_effect(df_slice):
                # Check the open price of the last candle in the slice, which is current candle `i`
                if df_slice.iloc[-1]['open'] == 110.5 and df_slice.iloc[-1]['close'] == 107.5: # Our specific bearish engulfing candle 10
                    return "bearishEngulfing"
                return None # No other candle is bearish engulfing
            mock_cnn_instance._detect_engulfing.side_effect = mock_detect_engulfing_side_effect

            # Simulate _detect_bull_flag behavior
            # This will be used to determine which candles (i) are considered bullFlags
            def mock_detect_bull_flag_side_effect(df_slice):
                idx = df_slice.index[-1] # The candle being evaluated is df.iloc[i]
                # BullFlags at original indices 9, 10, 11
                if idx == dummy_df.index[9] or idx == dummy_df.index[10] or idx == dummy_df.index[11]:
                    return "bullFlag"
                return None
            mock_cnn_instance._detect_bull_flag.side_effect = mock_detect_bull_flag_side_effect

            # Call the method under test
            prepared_datasets = await pre_trainer.prepare_training_data(dummy_df)

        # Assertions
        assert "bullFlag" in prepared_datasets
        assert "bearishEngulfing" in prepared_datasets

        bull_flag_df = prepared_datasets["bullFlag"]
        bearish_engulfing_df = prepared_datasets["bearishEngulfing"]

        # Check expected labels for bullFlag
        # Candles 0, 1, 2 were marked as bullFlag by the mock.
        # Recall: future_lookahead_bull = 3, price_increase_threshold_bull = 0.02
        # Candle 0: close=101. Future close (idx 3) = 104. (104-101)/101 = 0.0297 > 0.02. Label 1.
        # Candle 1: close=101. Future close (idx 4) = 104. (104-101)/101 = 0.0297 > 0.02. Label 1.
        # Candle 2: close=102. Future close (idx 5) = 105. (105-102)/102 = 0.0294 > 0.02. Label 1.

        # The returned df from prepare_training_data contains sequences.
        # If sequence_length_bull = 10, the first possible labeled data point in bull_flag_df
        # would correspond to the original candle index 9 (if it was a bullflag).
        # The 'label' is for the candle at the END of the sequence.
        # Let's re-check how labels are assigned in relation to sequences.
        # The loop is `for i in range(sequence_length -1, len(df) - future_lookahead)`
        # `current_pattern = self.cnn_patterns_labeling._detect_bull_flag(df.iloc[i:i+1])` -> This is wrong.
        # It should be `df.iloc[i - (sequence_length-1) : i+1]` for pattern detection on sequence.
        # Let's assume pattern detection is on a single candle for now as per `_detect_bull_flag(df.iloc[i:i+1])`.
        # The label is for `df['close'].iloc[i]`.

        # The output df `bull_flag_df` will have rows where each row is a sequence,
        # and the label corresponds to the pattern and future profit of the *last* candle in that sequence.

        # For bullFlag, given sequence_length_bull=10 (default from _get_param_side_effect):
        # The first possible *sequence* ends at index 9. If candle 9 was a bullFlag, it would be labeled.
        # Our mock marks candles 0, 1, 2 as bullFlag. These are too early to form a full sequence of 10
        # if the sequence must END at these points for labeling.

        # Let's adjust which candles are bullFlag for easier testing with sequences.
        # Let candle 9, 10, 11 be bullFlags.
        # sequence_length_bull = 10.
        # Candle 9 (ends sequence 0-9): close=110. Future (9+3=12) close=105 (original was 112, changed for bearish).
        # Original data['close'][12] = 112. (112-110)/110 = 2/110 = 0.018. This is < 0.02. Label 0.
        # Candle 10 (ends sequence 1-10): close=107.5. Future (10+3=13) close=111. (111-107.5)/107.5 = 3.5/107.5 = 0.032 > 0.02. Label 1.
        # Candle 11 (ends sequence 2-11): close=112. Future (11+3=14) close=113. (113-112)/112 = 1/112 = 0.0089 < 0.02. Label 0.

        # Re-mock _detect_bull_flag:
        def new_mock_detect_bull_flag(df_slice): # df_slice is df.iloc[i:i+1]
            idx = df_slice.index[-1]
            if idx == dummy_df.index[9] or idx == dummy_df.index[10] or idx == dummy_df.index[11]:
                return "bullFlag"
            return None
        mock_cnn_instance._detect_bull_flag.side_effect = new_mock_detect_bull_flag

        # Re-run with updated mock for bull flags
        # Must re-patch as the context manager exited
        with patch('core.pre_trainer.CnnPatternsForLabeling.instance') as MockCnnPatternsInstance: # If it's a singleton
             # If CnnPatternsForLabeling is instantiated per call, the original patch is fine.
             # PreTrainer creates `self.cnn_patterns_labeling = CnnPatternsForLabeling(self.params_manager)`
             # So the original patch on the class itself is correct.
            pass # This re-run logic is getting complicated, let's simplify the test setup first.

        # For now, let's assume the first call to prepare_training_data was with the intended logic.
        # The key is to ensure the *output* dataframe's labels are correct based on *input* data
        # and mocked pattern detection.

        # The `prepared_datasets` should contain sequences.
        # If candle 10 was detected as "bearishEngulfing" (it was, by mock_detect_engulfing for open=110.5)
        # and sequence_length_bear = 10 (default from _get_param_side_effect)
        # and future_lookahead_bear = 2, price_decrease_threshold_bear = 0.01
        # The sequence is data[1:11] (ending at index 10).
        # Label for candle 10 (close=107.5):
        # Future close (10+2=12) is 105.
        # Profit: (107.5 - 105) / 107.5 = 2.5 / 107.5 = 0.023 > 0.01. So label should be 1.
        # The `bearish_engulfing_df` should have one row (sequence ending at index 10), and its label should be 1.
        assert len(bearish_engulfing_df) >= 1
        # Find the row corresponding to candle 10. The 'close' value in the last row of features should be 107.5
        # The features are scaled, so direct comparison is hard.
        # Instead, rely on the fact that only one bearishEngulfing was "detected".
        assert bearish_engulfing_df['label'].iloc[0] == 1 # Assuming it's the first and only one

        # For bullFlag:
        # Mock detected bullFlag at index 9, 10, 11.
        # sequence_length_bull = 10.
        # Candle 9 (sequence 0-9), close=110. Future (9+3=12) close=105. (105-110)/110 = -0.045. Not > 0.02. Label 0.
        # Candle 10 (sequence 1-10), close=107.5. Future (10+3=13) close=111. (111-107.5)/107.5 = 0.032 > 0.02. Label 1.
        # Candle 11 (sequence 2-11), close=112. Future (11+3=14) close=113. (113-112)/112 = 0.0089 < 0.02. Label 0.

        # Check which sequences are actually generated:
        # Loop is `for i in range(sequence_length - 1, len(df) - future_lookahead)`
        # Min index `i` for bullFlag (seq_len=10, lookahead=3): `range(9, 18-3=15)` -> i = 9,10,11,12,13,14
        # Candles that will be checked for bullFlag pattern: 9, 10, 11, 12, 13, 14
        # Our mock returns "bullFlag" for 9, 10, 11.

        # Expected labels in bull_flag_df (should have 3 rows):
        # Row 0 (from candle 9): Label 0
        # Row 1 (from candle 10): Label 1
        # Row 2 (from candle 11): Label 0
        assert len(bull_flag_df) == 3 # Three bullflags detected and processable
        assert bull_flag_df['label'].iloc[0] == 0 # Corresponds to original candle 9
        assert bull_flag_df['label'].iloc[1] == 1 # Corresponds to original candle 10
        assert bull_flag_df['label'].iloc[2] == 0 # Corresponds to original candle 11


    def _get_param_side_effect(self, param_name, strategy_id=None):
        # Parameters used in PreTrainer.prepare_training_data and train_ai_models
        # These names must match how they are constructed in pre_trainer.py (e.g., f"dataSequenceLength{pattern_name_cap}")

        # For pattern "bullFlag" (pattern_name_cap = "BullFlag")
        if param_name == "dataSequenceLengthBullFlag": return 10
        if param_name == "priceIncreaseThresholdBullFlag": return 0.02
        if param_name == "futureLookaheadBullFlag": return 3
        if param_name == "maxDropThresholdBullFlag": return 0.05

        # For pattern "bearishEngulfing" (pattern_name_cap = "BearishEngulfing")
        if param_name == "dataSequenceLengthBearishEngulfing": return 10
        if param_name == "priceDecreaseThresholdBearishEngulfing": return 0.01
        if param_name == "futureLookaheadBearishEngulfing": return 2
        if param_name == "maxRiseThresholdBearishEngulfing": return 0.05

        if param_name == "patternsToPretrain": return ["bullFlag", "bearishEngulfing"]

        # Params for train_ai_models part (not pattern specific in getter)
        if param_name == "numEpochsPretrain": return 1 # Keep low for testing
        if param_name == "learningRatePretrain": return 0.001
        if param_name == "modelSaveDir": return MODELS_DIR
        if param_name == "pretrainLogFile": return PRETRAIN_LOG_FILE
        return None

    @pytest.mark.asyncio
    @patch('core.pre_trainer.torch.save')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(PreTrainer, '_log_pretrain_activity', new_callable=AsyncMock)
    @patch('core.pre_trainer.SimpleCNN')
    async def test_pre_trainer_train_ai_models_training_and_saving(
        self,
        MockSimpleCNN, # Patched object from @patch('core.pre_trainer.SimpleCNN')
        mock_log_activity, # Patched object from @patch.object(PreTrainer, '_log_pretrain_activity'...)
        mock_file_open, # Patched object from @patch('builtins.open'...)
        mock_torch_save # Patched object from @patch('core.pre_trainer.torch.save')
    ):
        mock_params_manager = MagicMock(spec=ParamsManager)
        mock_params_manager.get_param.side_effect = self._get_param_side_effect
        mock_cooldown_tracker = MagicMock(spec=CooldownTracker)

        pre_trainer = PreTrainer(params_manager=mock_params_manager, cooldown_tracker=mock_cooldown_tracker)

        # Create dummy prepared_datasets
        # These DataFrames should have features and a 'label' column, as produced by prepare_training_data
        # The actual content of feature columns doesn't matter as SimpleCNN is mocked,
        # but column names and structure should be representative for scaler fitting.
        # sequence_length is 10 (from _get_param_side_effect). DataFrames need enough rows.
        raw_bull_features = pd.DataFrame({
            'open': range(100, 115), 'high': range(100, 115), 'low': range(100, 115),
            'close': range(101, 116), 'volume': range(1000, 1015), # 15 data points
            'label': [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        })
        raw_bear_features = pd.DataFrame({
            'open': range(200, 212), 'high': range(200, 212), 'low': range(200, 212),
            'close': range(201, 213), 'volume': range(1200, 1212), # 12 data points
            'label': [1,0,1,0,1,0,1,0,1,0,1,0]
        })

        prepared_datasets_mock = {
            "bullFlag": raw_bull_features,
            "bearishEngulfing": raw_bear_features
        }

        # Mock the SimpleCNN instance and its train_model method (or equivalent)
        mock_cnn_model_instance = MagicMock()
        # SimpleCNN(input_dim, hidden_dim, num_layers, output_dim) -> constructor of SimpleCNN
        # train_ai_models gets these from feature shape after scaling and sequencing
        # For now, MockSimpleCNN.return_value covers the instance.
        # If SimpleCNN constructor is called with specific args derived from data,
        # we might need to check those if they were complex. But train_ai_models creates it.
        MockSimpleCNN.return_value = mock_cnn_model_instance

        # Mock os.makedirs which is called inside train_ai_models before saving
        with patch('os.makedirs', new_callable=MagicMock) as mock_makedirs:
            await pre_trainer.train_ai_models(prepared_datasets_mock)
            # Assert that makedirs was called for the base model directory
            mock_makedirs.assert_any_call(MODELS_DIR, exist_ok=True)

        # Assertions for torch.save
        expected_model_path_bull = os.path.join(MODELS_DIR, "cnn_model_bullFlag.pth")
        expected_model_path_bear = os.path.join(MODELS_DIR, "cnn_model_bearishEngulfing.pth")

        # Check that torch.save was called with model.state_dict() and the correct path
        # Call args are like call(model.state_dict(), path)
        # We check the path argument here.
        saved_model_paths = [c.args[1] for c in mock_torch_save.call_args_list]
        assert expected_model_path_bull in saved_model_paths
        assert expected_model_path_bear in saved_model_paths
        assert mock_torch_save.call_count == 2


        # Assertions for json.dump (via mock_file_open for scaler parameters)
        expected_scaler_path_bull = os.path.join(MODELS_DIR, "scaler_params_bullFlag.json")
        expected_scaler_path_bear = os.path.join(MODELS_DIR, "scaler_params_bearishEngulfing.json")

        # Check that 'open' was called with the correct scaler file paths in write mode
        opened_files_for_writing = []
        for call_args in mock_file_open.call_args_list:
            if call_args[0][0] in [expected_scaler_path_bull, expected_scaler_path_bear] and call_args[0][1] == 'w':
                opened_files_for_writing.append(call_args[0][0])

        assert expected_scaler_path_bull in opened_files_for_writing
        assert expected_scaler_path_bear in opened_files_for_writing
        # This also implies json.dump would be called with the file handle from open()

        # Assertions for _log_pretrain_activity
        assert mock_log_activity.call_count == 2
        mock_log_activity.assert_any_call(
            pattern_name="bullFlag",
            action="model_trained_and_saved",
            details=f"Model and scaler saved to {MODELS_DIR}"
        )
        mock_log_activity.assert_any_call(
            pattern_name="bearishEngulfing",
            action="model_trained_and_saved",
            details=f"Model and scaler saved to {MODELS_DIR}"
        )

        # Assert that SimpleCNN was instantiated
        assert MockSimpleCNN.call_count == 2 # Once per pattern

        # Assert that the training method on the CNN instance was called
        # Assuming the method is 'train_model'. If it's different, this needs to change.
        # Example: mock_cnn_model_instance.train.call_count if method is 'train'
        assert mock_cnn_model_instance.train_model.call_count == 2

        # Verify that get_param was called for relevant training parameters
        mock_params_manager.get_param.assert_any_call("numEpochsPretrain")
        mock_params_manager.get_param.assert_any_call("learningRatePretrain")
        mock_params_manager.get_param.assert_any_call("modelSaveDir")
        mock_params_manager.get_param.assert_any_call("pretrainLogFile")
        mock_params_manager.get_param.assert_any_call("dataSequenceLengthBullFlag") # Used in _create_sequences_and_scale
        mock_params_manager.get_param.assert_any_call("dataSequenceLengthBearishEngulfing") # Used in _create_sequences_and_scale

# New imports for CNNPatterns tests
from core.cnn_patterns import CNNPatterns
# SimpleCNN is defined inside cnn_patterns.py, so for patching it's 'core.cnn_patterns.SimpleCNN'
# MinMaxScaler is from sklearn, so for patching it's 'core.cnn_patterns.MinMaxScaler'
# as it's imported directly in cnn_patterns.py

TEST_CNN_MODELS_DIR = "test_data/cnn_models_pattern_detection"

# New Test Class for CNNPatterns
class TestCNNPatterns:
    @pytest.mark.asyncio
    @patch('core.cnn_patterns.os.path.exists')
    @patch('core.cnn_patterns.torch.load')
    @patch('core.cnn_patterns.SimpleCNN')
    @patch('core.cnn_patterns.json.load')
    @patch('core.cnn_patterns.MinMaxScaler')
    @patch('core.cnn_patterns.CNNPatterns.MODELS_DIR', TEST_CNN_MODELS_DIR)
    async def test_cnn_patterns_model_scaler_loading(
        self,
        MockMinMaxScaler,
        mock_json_load,
        MockSimpleCNNClass, # Patched class 'core.cnn_patterns.SimpleCNN'
        mock_torch_load,
        mock_os_path_exists
    ):
        # Configure mocks
        mock_os_path_exists.return_value = True

        mock_model_state_dict = {'layer.weight': torch.randn(1,1)}
        mock_torch_load.return_value = mock_model_state_dict

        dummy_scaler_params = {
            'min_': np.array([0.1] * 9).tolist(), # np.array to list for direct comparison
            'scale_': np.array([0.9] * 9).tolist(),
            'feature_names_in_': ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist'],
            'sequence_length': 30
        }
        mock_json_load.return_value = dummy_scaler_params

        mock_simple_cnn_instance = MagicMock(spec=MockSimpleCNNClass)
        MockSimpleCNNClass.return_value = mock_simple_cnn_instance

        mock_scaler_instance = MagicMock(spec=MinMaxScaler)
        # When cnn_patterns.py calls MinMaxScaler(), this instance will be returned.
        # We then need to allow attributes to be set on it.
        mock_scaler_instance.min_ = None
        mock_scaler_instance.scale_ = None
        mock_scaler_instance.n_features_in_ = 0
        mock_scaler_instance.feature_names_in_ = None
        MockMinMaxScaler.return_value = mock_scaler_instance

        # Instantiate CNNPatterns - this will trigger _load_cnn_models_and_scalers
        cnn_patterns = CNNPatterns()

        # Assertions
        expected_patterns = list(cnn_patterns.pattern_configs.keys()) # Use keys from instance

        assert set(cnn_patterns.models.keys()) == set(expected_patterns)
        for pattern_name_iter in expected_patterns: # Renamed to avoid conflict with outer scope 'pattern_name'
            assert cnn_patterns.models[pattern_name_iter] == mock_simple_cnn_instance

            # Verify SimpleCNN was instantiated with correct params
            # Check if any of the calls to SimpleCNN constructor match this pattern's config
            call_matched = False
            for call_obj in MockSimpleCNNClass.call_args_list:
                if (call_obj.kwargs['input_channels'] == cnn_patterns.pattern_configs[pattern_name_iter]['input_channels'] and
                    call_obj.kwargs['num_classes'] == cnn_patterns.pattern_configs[pattern_name_iter]['num_classes'] and
                    call_obj.kwargs['sequence_length'] == dummy_scaler_params['sequence_length']):
                    call_matched = True
                    break
            assert call_matched, f"SimpleCNN constructor not called with expected args for {pattern_name_iter}"

            # Verify model.load_state_dict and model.eval were called on the instance
            # Since it's the same mock instance for all, check total calls
        assert mock_simple_cnn_instance.load_state_dict.call_count == len(expected_patterns)
        mock_simple_cnn_instance.load_state_dict.assert_any_call(mock_model_state_dict)
        assert mock_simple_cnn_instance.eval.call_count == len(expected_patterns)


        assert set(cnn_patterns.scalers.keys()) == set(expected_patterns)
        for pattern_name_iter in expected_patterns:
            loaded_scaler = cnn_patterns.scalers[pattern_name_iter]
            assert loaded_scaler == mock_scaler_instance

            # Verify attributes were set on the mock_scaler_instance
            # These are set directly in _load_cnn_models_and_scalers
            # Convert to list for comparison if original is numpy array
            assert loaded_scaler.min_.tolist() == dummy_scaler_params['min_']
            assert loaded_scaler.scale_.tolist() == dummy_scaler_params['scale_']
            assert loaded_scaler.n_features_in_ == len(dummy_scaler_params['feature_names_in_'])
            assert loaded_scaler.feature_names_in_.tolist() == dummy_scaler_params['feature_names_in_']

        # Verify torch.load and json.load calls with correct paths
        for pattern_name_config, config_dict in cnn_patterns.pattern_configs.items(): # Renamed
            expected_model_path = os.path.join(TEST_CNN_MODELS_DIR, os.path.basename(config_dict['model_path']))
            # Scaler path for open() call verification if we were mocking open for json.load
            # expected_scaler_path = os.path.join(TEST_CNN_MODELS_DIR, os.path.basename(config_dict['scaler_path']))

            mock_torch_load.assert_any_call(expected_model_path)
            # mock_open (if used for json.load) would be checked with expected_scaler_path

        assert mock_json_load.call_count == len(expected_patterns)
        assert mock_torch_load.call_count == len(expected_patterns)
        assert MockMinMaxScaler.call_count == len(expected_patterns)
        assert MockSimpleCNNClass.call_count == len(expected_patterns)

    @pytest.mark.asyncio
    @patch.object(CNNPatterns, '_load_cnn_models_and_scalers') # Prevent actual loading
    async def test_cnn_patterns_predict_pattern_score_output(self, mock_load_models_scalers):
        # Instantiate CNNPatterns; _load_cnn_models_and_scalers is mocked, so models/scalers are empty
        cnn_patterns = CNNPatterns()

        # Create a mock SimpleCNN model instance
        mock_model_instance = MagicMock() # spec=SimpleCNN if SimpleCNN was imported from core.cnn_patterns
        # The forward method should return a tensor of shape (batch_size, num_classes)
        # e.g., for batch_size=1, num_classes=2: tensor([[0.2, 0.8]])
        # torch.softmax will be applied to this output.
        mock_model_instance.forward.return_value = torch.tensor([[0.2345, 0.7655]]) # Logits

        # Create a mock MinMaxScaler instance
        mock_scaler_instance = MagicMock() # spec=MinMaxScaler if imported
        # The transform method should return a NumPy array of shape (sequence_length, num_features)
        # num_features = 9 (open, high, low, close, volume, rsi, macd, macdsignal, macdhist)
        # sequence_length = 30 (default in _dataframe_to_cnn_input)
        # So, shape (30, 9)
        dummy_scaled_data = np.random.rand(30, 9)
        mock_scaler_instance.transform.return_value = dummy_scaled_data
        # Also provide feature_names_in_ as it's used in _dataframe_to_cnn_input for safety checks
        mock_scaler_instance.feature_names_in_ = np.array(['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist'])


        # Directly populate the models and scalers dictionaries for a 'testPattern'
        test_pattern_name = 'testPattern'
        cnn_patterns.models[test_pattern_name] = mock_model_instance
        cnn_patterns.scalers[test_pattern_name] = mock_scaler_instance

        # Create a dummy Pandas DataFrame
        # It needs at least `sequence_length` (default 30) rows.
        # Columns should include those in features_cols of _dataframe_to_cnn_input
        num_rows = 35
        features_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist']
        data = {col: np.random.rand(num_rows) for col in features_cols}
        mock_df = pd.DataFrame(data)
        # Add a 'date' column as _dataframe_to_candles (called by multi-tf detector, not directly by predict_pattern_score)
        # might expect it. It's not strictly needed by _dataframe_to_cnn_input if 'date' is not a feature.
        # mock_df['date'] = pd.to_datetime([datetime.utcnow() - timedelta(minutes=i) for i in range(num_rows)])

        # Call predict_pattern_score
        score = await cnn_patterns.predict_pattern_score(mock_df, test_pattern_name)

        # Assertions
        assert isinstance(score, float), "Score should be a float"
        assert 0.0 <= score <= 1.0, f"Score {score} should be between 0.0 and 1.0"

        # Expected score from torch.softmax(torch.tensor([[0.2345, 0.7655]]), dim=1)[0, 1]
        expected_probabilities = torch.softmax(torch.tensor([[0.2345, 0.7655]]), dim=1)
        expected_score = expected_probabilities[0, 1].item()
        assert abs(score - expected_score) < 1e-6, f"Score {score} did not match expected {expected_score}"


        # Verify that the model's forward method and scaler's transform method were called
        mock_model_instance.forward.assert_called_once()
        mock_scaler_instance.transform.assert_called_once()

        # Verify the input to forward was a tensor of the correct shape
        # Input to forward is (1, num_features, sequence_length) -> (1, 9, 30)
        # after permute(0,2,1) from (1, sequence_length, num_features)
        call_args_model_forward = mock_model_instance.forward.call_args[0][0]
        assert isinstance(call_args_model_forward, torch.Tensor)
        assert call_args_model_forward.shape == (1, 9, 30) # (batch_size, input_channels, sequence_length)

        # Verify the input to scaler.transform was a numpy array of correct shape (sequence_length, num_features)
        call_args_scaler_transform = mock_scaler_instance.transform.call_args[0][0]
        assert isinstance(call_args_scaler_transform, np.ndarray)
        assert call_args_scaler_transform.shape == (30, 9)

# Imports for EntryDecider tests
from core.entry_decider import EntryDecider
# Dependencies of EntryDecider that need mocking:
# GPTReflector, GrokReflector, PromptBuilder, BiasReflector,
# ConfidenceEngine, CNNPatterns are already imported or accessible via core.entry_decider.
# ParamsManager, CooldownTracker are already imported.
# datetime is already imported.
# pandas is already imported (as pd).


# Define a helper function for ParamsManager mock side effects for EntryDecider tests
def get_entry_decider_param_side_effect(scenario_params):
    """
    Returns a side_effect function for mock_params_manager.get_param
    based on the provided scenario parameters.
    """
    def side_effect(key, strategy_id=None, default=None):
        # Global params or those not strategy-specific can be here
        if key == "timeOfDayEffectiveness":
            # Default to neutral unless specified in scenario_params
            return scenario_params.get(key, {str(datetime.now().hour % 24).zfill(2): 0.0})

        # Fallback to strategy-specific, then general scenario_params, then global defaults from EntryDecider itself
        val = scenario_params.get(f"{strategy_id}_{key}", scenario_params.get(key, default))

        # Log the parameter being fetched and its returned value for debugging tests
        # print(f"ParamsManager.get_param(key='{key}', strategy_id='{strategy_id}') -> returning: {val}, default was: {default}")
        return val
    return side_effect

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_name, mock_params, mock_cnn_patterns_output, ai_consensus_intent, ai_combined_confidence, bias_score, expected_enter, expected_reason_part, learned_bias_override, learned_confidence_override, entry_conviction_override",
    [
        # Scenario 1: Entry Approved - Strong Pattern Threshold Met
        (
            "Approve_StrongPatternMet",
            { # mock_params for ParamsManager
                "cnnPatternWeight": 0.8,
                "entryRulePatternScore": 0.0, # No rule-based pattern score for this specific sub-scenario
                "strongPatternThreshold": 0.7,
                "entryConvictionThreshold": 0.6, # This will be overridden by argument to should_enter for clarity
                "entryLearnedBiasThreshold": 0.5,
                "entryTimeEffectivenessImpactFactor": 0.5,
                # timeOfDayEffectiveness will be neutral by default from get_entry_decider_param_side_effect
            },
            {"cnn_predictions": {"15m_bullFlag_score": 0.95}, "patterns": {}}, # CNN score 0.95; 0.95*0.8 = 0.76 >= 0.7 (strong)
            "LONG", # ai_consensus_intent
            0.75,   # ai_combined_confidence (0.75 >= 0.6 entry_conviction_threshold)
            0.55,   # bias_score (0.55 >= 0.5 entryLearnedBiasThreshold)
            True,   # expected_enter
            "AI_CONSENSUS_LONG_CONDITIONS_MET", # expected_reason_part
            0.6,    # learned_bias_override
            0.75,   # learned_confidence_override
            0.6     # entry_conviction_override for should_enter
        ),
        # Scenario 2: Entry Denied - Strong Pattern Threshold NOT Met
        (
            "Deny_StrongPatternNotMet",
            { # mock_params for ParamsManager
                "cnnPatternWeight": 0.8,
                "entryRulePatternScore": 0.0,
                "strongPatternThreshold": 0.6,
                "entryConvictionThreshold": 0.5, # Overridden by arg
                "entryLearnedBiasThreshold": 0.5,
                "entryTimeEffectivenessImpactFactor": 0.5,
            },
            {"cnn_predictions": {"15m_bullFlag_score": 0.70}, "patterns": {}}, # CNN score 0.70; 0.70*0.8 = 0.56 < 0.6 (not strong)
            "LONG", # ai_consensus_intent
            0.9,    # ai_combined_confidence (e.g. 0.9 >= 0.5 entry_conviction_threshold)
            0.55,   # bias_score (0.55 >= 0.5 entryLearnedBiasThreshold)
            False,  # expected_enter
            "pattern_score_low (0.56_vs_0.60)", # expected_reason_part
            0.6,    # learned_bias_override
            0.9,    # learned_confidence_override
            0.5     # entry_conviction_override
        ),
    ]
)
@patch('core.entry_decider.GPTReflector')
@patch('core.entry_decider.GrokReflector')
@patch('core.entry_decider.PromptBuilder')
@patch('core.entry_decider.BiasReflector')
@patch('core.entry_decider.ConfidenceEngine') # Assuming it's used, though not directly in plan
@patch('core.entry_decider.ParamsManager')
@patch('core.entry_decider.CooldownTracker')
@patch('core.entry_decider.CNNPatterns')
async def test_entry_decider_should_enter(
    MockCNNPatterns, MockCooldownTracker, MockParamsManager, MockConfidenceEngine,
    MockBiasReflector, MockPromptBuilder, MockGrokReflector, MockGPTReflector,
    scenario_name, mock_params, mock_cnn_patterns_output, ai_consensus_intent,
    ai_combined_confidence, bias_score, expected_enter, expected_reason_part,
    learned_bias_override, learned_confidence_override, entry_conviction_override
):
    # Setup mocks
    mock_params_manager_instance = MockParamsManager.return_value
    mock_params_manager_instance.get_param.side_effect = get_entry_decider_param_side_effect(mock_params)

    mock_cooldown_tracker_instance = MockCooldownTracker.return_value
    mock_cooldown_tracker_instance.is_cooldown_active.return_value = False

    mock_cnn_patterns_instance = MockCNNPatterns.return_value
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.return_value = mock_cnn_patterns_output
    # For these specific tests, we don't need to mock get_all_detectable_pattern_keys
    # as the default implementation in CNNPatterns should suffice if it returns ['bullFlag', 'bearishEngulfing']
    # and EntryDecider calls it. If tests need specific control, it can be added to the mock_cnn_patterns_instance.
    # For now, assume EntryDecider correctly calls the real method on the mocked instance.

    # Mock AI consensus parts
    # GPTReflector and GrokReflector are part of get_consensus, so their ask_ai/ask_grok need mocking
    # The EntryDecider.get_consensus method itself can be mocked for simpler setup if details of GPT/Grok not needed
    # For now, let's assume get_consensus is called and we mock its constituents or itself.
    # The plan implies mocking AI consensus to return specific intent and confidence.
    # We can patch EntryDecider.get_consensus directly.

    mock_get_consensus_result = {
        "consensus_intentie": ai_consensus_intent,
        "combined_confidence": ai_combined_confidence,
        "combined_bias_reported": 0.5, # Default, not primary focus of these scenarios
        "gpt_raw": {}, "grok_raw": {}
    }

    mock_bias_reflector_instance = MockBiasReflector.return_value
    mock_bias_reflector_instance.get_bias_score.return_value = bias_score

    # PromptBuilder mock (if generate_prompt_with_data is called)
    mock_prompt_builder_instance = MockPromptBuilder.return_value
    mock_prompt_builder_instance.generate_prompt_with_data.return_value = "Test prompt"

    # Instantiate EntryDecider
    # Patching EntryDecider.get_consensus directly after instantiation or via @patch.object
    # Make sure EntryDecider's __init__ doesn't have complex dependencies not mocked here
    # For now, assuming ParamsManager, PromptBuilder, GPTReflector, GrokReflector are the main ones.
    # If BiasReflector, ConfidenceEngine are also initialized in __init__, they might need mocking too.

    # It's crucial that EntryDecider uses the *instance* of CNNPatterns that we are configuring.
    # So, we pass mock_cnn_patterns_instance to the EntryDecider constructor if possible,
    # or ensure the patch applies to the instance used by EntryDecider.
    # The current patching approach `patch('core.entry_decider.CNNPatterns', return_value=mock_cnn_patterns_instance)`
    # should mean that when EntryDecider does `self.cnn_patterns_detector = CNNPatterns()`, it gets our mock.

    with patch('core.entry_decider.ParamsManager', return_value=mock_params_manager_instance), \
         patch('core.entry_decider.PromptBuilder', return_value=mock_prompt_builder_instance), \
         patch('core.entry_decider.GPTReflector', MockGPTReflector), \
         patch('core.entry_decider.GrokReflector', MockGrokReflector), \
         patch('core.entry_decider.BiasReflector', MockBiasReflector), \
         patch('core.entry_decider.ConfidenceEngine', MockConfidenceEngine), \
         patch('core.entry_decider.CNNPatterns', return_value=mock_cnn_patterns_instance), \
         patch('core.entry_decider.CooldownTracker', return_value=mock_cooldown_tracker_instance):
        entry_decider = EntryDecider() # This will now use the mocked CNNPatterns instance
        entry_decider.get_consensus = AsyncMock(return_value=mock_get_consensus_result)


    # Prepare inputs for should_enter
    symbol = "BTC/USDT"
    current_strategy_id = "TestStrategy"

    # Create a dummy base DataFrame and candles_by_timeframe
    # Ensure the DataFrame has 'timeframe' attribute if EntryDecider logic uses it.
    # The CNN score key '15m_bullFlag_score' implies a 15m timeframe is relevant.
    mock_df_15m = pd.DataFrame({
        'open': np.random.rand(50), 'high': np.random.rand(50), 'low': np.random.rand(50),
        'close': np.random.rand(50), 'volume': np.random.rand(50)
    })
    mock_df_15m.attrs['timeframe'] = '15m' # Important for constructing CNN score key if dynamic

    candles_by_timeframe = {'15m': mock_df_15m}
    trade_context = {'candles_by_timeframe': candles_by_timeframe, 'current_price': 100.0}

    # Call should_enter
    decision = await entry_decider.should_enter(
        dataframe=mock_df_15m, # Base dataframe
        symbol=symbol,
        current_strategy_id=current_strategy_id,
        trade_context=trade_context,
        learned_bias=learned_bias_override, # Use scenario specific override
        learned_confidence=learned_confidence_override, # Use scenario specific override
        entry_conviction_threshold=entry_conviction_override # Use scenario specific override
    )

    # Assertions
    assert decision['enter'] == expected_enter, f"Scenario '{scenario_name}': Enter decision mismatch. Reason: {decision.get('reason')}"
    if expected_reason_part:
        assert expected_reason_part in decision['reason'], f"Scenario '{scenario_name}': Expected reason part '{expected_reason_part}' not in actual reason '{decision['reason']}'"

    # Verify get_param calls for specific params related to the logic being tested
    # These checks can be very specific if needed by iterating call_args_list
    mock_params_manager_instance.get_param.assert_any_call("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
    mock_params_manager_instance.get_param.assert_any_call("strongPatternThreshold", strategy_id=current_strategy_id, default=0.5)
    # Removed condition for checking pattern_score_low as it's too specific for a general param check section

    if expected_enter or ("final_ai_conf_low" not in decision['reason'] and "ai_intent_not_long" not in decision['reason']):
        # Check bias threshold if other conditions didn't lead to early exit from check
        mock_params_manager_instance.get_param.assert_any_call("entryLearnedBiasThreshold", strategy_id=current_strategy_id, default=0.55)

    # Verify CNNPatterns.detect_patterns_multi_timeframe was called
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.assert_called_once_with(candles_by_timeframe, symbol)

# --- Start of new test function for CNN pattern weighting ---

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_name, mock_params_config, cnn_predictions_config, rule_patterns_config, expected_weighted_score, expected_log_parts, mock_detectable_keys",
    [
        (
            "Multi_CNN_IndividualWeights",
            { # mock_params_config
                "cnn_bullFlag_weight": 0.7, "cnn_bearishEngulfing_weight": 1.2, "cnnPatternWeight": 1.0, # Fallback
                "entryRulePatternScore": 0.5, "strongPatternThreshold": 0.1, "entryConvictionThreshold": 0.5,
                "entryLearnedBiasThreshold": 0.5, "entryTimeEffectivenessImpactFactor": 0.0
            },
            {"5m_bullFlag_score": 0.8, "1h_bearishEngulfing_score": 0.9}, # cnn_predictions_config
            {}, # rule_patterns_config
            (0.8 * 0.7) + (0.9 * 1.2), # expected_weighted_score = 0.56 + 1.08 = 1.64
            ["Using specific weight for CNN pattern 'bullFlag': 0.7",
             "Using specific weight for CNN pattern 'bearishEngulfing': 1.2",
             "CNN Contribution: Pattern 'bullFlag' (from key '5m_bullFlag_score') Score: 0.80 * Weight: 0.70 = 0.56",
             "CNN Contribution: Pattern 'bearishEngulfing' (from key '1h_bearishEngulfing_score') Score: 0.90 * Weight: 1.20 = 1.08",
             "Final Calculated Weighted Pattern Score: 1.64"],
            ['bullFlag', 'bearishEngulfing'] # mock_detectable_keys
        ),
        (
            "CNN_Uses_FallbackWeight",
            { # mock_params_config: cnn_bullFlag_weight is missing, so fallback cnnPatternWeight=0.9 is used.
                "cnnPatternWeight": 0.9, "entryRulePatternScore": 0.0, "strongPatternThreshold": 0.1,
                "entryConvictionThreshold": 0.5, "entryLearnedBiasThreshold": 0.5, "entryTimeEffectivenessImpactFactor": 0.0
            },
            {"5m_bullFlag_score": 0.8}, # cnn_predictions_config
            {}, # rule_patterns_config
            0.8 * 0.9, # expected_weighted_score = 0.72
            ["Using fallback cnnPatternWeight (0.9) for CNN pattern 'bullFlag'",
             "CNN Contribution: Pattern 'bullFlag' (from key '5m_bullFlag_score') Score: 0.80 * Weight: 0.90 = 0.72",
             "Final Calculated Weighted Pattern Score: 0.72"],
            ['bullFlag']
        ),
        (
            "CNN_and_RuleBased_Combination",
            { # mock_params_config
                "cnn_bullFlag_weight": 1.1, "cnnPatternWeight": 1.0, # Fallback for rule pattern
                "entryRulePatternScore": 0.5, "strongPatternThreshold": 0.1, "entryConvictionThreshold": 0.5,
                "entryLearnedBiasThreshold": 0.5, "entryTimeEffectivenessImpactFactor": 0.0
            },
            {"5m_bullFlag_score": 0.7}, # cnn_predictions_config
            {"bullishEngulfing": True}, # rule_patterns_config
            (0.7 * 1.1) + (0.5 * 1.0), # expected_weighted_score = 0.77 + 0.5 = 1.27
            ["Using specific weight for CNN pattern 'bullFlag': 1.1",
             "CNN Contribution: Pattern 'bullFlag' (from key '5m_bullFlag_score') Score: 0.70 * Weight: 1.10 = 0.77",
             "Detected rule-based pattern: 'bullishEngulfing'. Contribution: 0.50 * 1.00 = 0.50",
             "Final Calculated Weighted Pattern Score: 1.27"],
            ['bullFlag']
        ),
        (
            "No_CNN_Only_RuleBased",
            { # mock_params_config
                "cnnPatternWeight": 1.0, "entryRulePatternScore": 0.6, "strongPatternThreshold": 0.1,
                "entryConvictionThreshold": 0.5, "entryLearnedBiasThreshold": 0.5, "entryTimeEffectivenessImpactFactor": 0.0
            },
            {}, # cnn_predictions_config
            {"morningStar": True}, # rule_patterns_config
            0.6 * 1.0, # expected_weighted_score = 0.6
            ["No contributing CNN patterns detected",
             "Detected rule-based pattern: 'morningStar'. Contribution: 0.60 * 1.00 = 0.60",
             "Final Calculated Weighted Pattern Score: 0.60"],
            ['bullFlag'] # Still need some detectable keys for the loop, even if no scores match
        ),
        (
            "Malformed_CNN_Keys_Skipped",
            { # mock_params_config
                "cnn_bullFlag_weight": 1.0, "cnnPatternWeight": 0.8, "strongPatternThreshold": 0.1,
                "entryConvictionThreshold": 0.5, "entryLearnedBiasThreshold": 0.5, "entryTimeEffectivenessImpactFactor": 0.0,
                "entryRulePatternScore": 0.0
            },
            {"5m_bullFlag_score": 0.9, "malformed_key": 0.8, "1h_unknownPattern_score": 0.7}, # cnn_predictions_config
            {}, # rule_patterns_config
            0.9 * 1.0, # expected_weighted_score from bullFlag only
            ["CNN Contribution: Pattern 'bullFlag' (from key '5m_bullFlag_score') Score: 0.90 * Weight: 1.00 = 0.90",
             # "Could not extract a recognized CNN pattern name from cnn_predictions key: 'malformed_key'", # This log is DEBUG
             # "Weight for extracted CNN pattern name 'unknownPattern' (from key '1h_unknownPattern_score') not found" # This might happen if unknownPattern not in mock_detectable_keys
             "Final Calculated Weighted Pattern Score: 0.90"],
            ['bullFlag', 'someOtherPattern'] # bullFlag is detectable, unknownPattern is not
        ),
    ]
)
@patch('core.entry_decider.GPTReflector')
@patch('core.entry_decider.GrokReflector')
@patch('core.entry_decider.PromptBuilder')
@patch('core.entry_decider.BiasReflector')
@patch('core.entry_decider.ConfidenceEngine')
@patch('core.entry_decider.ParamsManager') # Patched at class level
@patch('core.entry_decider.CooldownTracker')
@patch('core.entry_decider.CNNPatterns') # Patched at class level
async def test_entry_decider_pattern_weighting_logic(
    MockCNNPatterns, MockCooldownTracker, MockParamsManager, MockConfidenceEngine,
    MockBiasReflector, MockPromptBuilder, MockGrokReflector, MockGPTReflector,
    scenario_name, mock_params_config, cnn_predictions_config, rule_patterns_config,
    expected_weighted_score, expected_log_parts, mock_detectable_keys, caplog
):
    caplog.set_level(logging.INFO) # Capture INFO level logs for assertions

    # Setup mocks for instances
    mock_params_manager_instance = MockParamsManager.return_value
    mock_params_manager_instance.get_param.side_effect = get_entry_decider_param_side_effect(mock_params_config)

    mock_cooldown_tracker_instance = MockCooldownTracker.return_value
    mock_cooldown_tracker_instance.is_cooldown_active.return_value = False

    mock_cnn_patterns_instance = MockCNNPatterns.return_value
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.return_value = {
        "cnn_predictions": cnn_predictions_config,
        "patterns": rule_patterns_config,
        "context": {} # Minimal context
    }
    # IMPORTANT: Mock get_all_detectable_pattern_keys
    mock_cnn_patterns_instance.get_all_detectable_pattern_keys.return_value = mock_detectable_keys

    # Mock AI consensus to be generally permissive for these tests focusing on pattern score
    mock_get_consensus_result = {
        "consensus_intentie": "LONG", "combined_confidence": 0.8, "combined_bias_reported": 0.5,
        "gpt_raw": {}, "grok_raw": {}
    }
    mock_bias_reflector_instance = MockBiasReflector.return_value # Not used directly, but init
    mock_bias_reflector_instance.get_bias_score.return_value = 0.6 # Permissive bias

    mock_prompt_builder_instance = MockPromptBuilder.return_value # Not used directly, but init
    mock_prompt_builder_instance.generate_prompt_with_data.return_value = "Test prompt for pattern weighting"

    # Instantiate EntryDecider
    with patch('core.entry_decider.ParamsManager', return_value=mock_params_manager_instance), \
         patch('core.entry_decider.PromptBuilder', return_value=mock_prompt_builder_instance), \
         patch('core.entry_decider.GPTReflector', MockGPTReflector), \
         patch('core.entry_decider.GrokReflector', MockGrokReflector), \
         patch('core.entry_decider.BiasReflector', return_value=mock_bias_reflector_instance), \
         patch('core.entry_decider.ConfidenceEngine', MockConfidenceEngine), \
         patch('core.entry_decider.CNNPatterns', return_value=mock_cnn_patterns_instance), \
         patch('core.entry_decider.CooldownTracker', return_value=mock_cooldown_tracker_instance):
        entry_decider = EntryDecider()
        entry_decider.get_consensus = AsyncMock(return_value=mock_get_consensus_result)


    # Prepare inputs for should_enter
    symbol = "TEST/COIN"
    current_strategy_id = "PatternWeightStrategy"
    mock_df_5m = pd.DataFrame({'open': [10]*60, 'high': [12]*60, 'low': [9]*60, 'close': [11]*60, 'volume': [100]*60})
    mock_df_5m.attrs['timeframe'] = '5m'
    candles_by_timeframe = {'5m': mock_df_5m, '1h': mock_df_5m.copy()} # Dummy data for other timeframes if needed
    trade_context = {'candles_by_timeframe': candles_by_timeframe, 'current_price': 11.0}

    # Call should_enter
    decision = await entry_decider.should_enter(
        dataframe=mock_df_5m, symbol=symbol, current_strategy_id=current_strategy_id,
        trade_context=trade_context,
        learned_bias=0.6, # Permissive
        learned_confidence=0.7, # Permissive
        entry_conviction_threshold=mock_params_config.get("entryConvictionThreshold", 0.5) # From scenario
    )

    # Assertions
    # Check weighted_pattern_score from the decision dict (it's returned on non-entry for logging)
    # or recalculate based on what should have happened.
    # For simplicity, we assume the test scenarios are set up so other conditions pass,
    # or we check the weighted_pattern_score logged.
    # The most reliable is to check the log for "Final Calculated Weighted Pattern Score".

    for log_part in expected_log_parts:
        assert log_part in caplog.text, f"Scenario '{scenario_name}': Expected log part '{log_part}' not found. Logs:\n{caplog.text}"

    # Find the final calculated score from logs, as it's the most direct way to verify the sum.
    # This regex will find "Final Calculated Weighted Pattern Score: <score>"
    import re
    final_score_match = re.search(r"Final Calculated Weighted Pattern Score: (\d+\.?\d*)", caplog.text)
    assert final_score_match is not None, f"Scenario '{scenario_name}': Could not find final weighted score in logs. Logs:\n{caplog.text}"
    logged_weighted_score = float(final_score_match.group(1))

    assert abs(logged_weighted_score - expected_weighted_score) < 0.001, \
        f"Scenario '{scenario_name}': Weighted pattern score mismatch. Expected {expected_weighted_score}, Got {logged_weighted_score}"

    # Verify that get_all_detectable_pattern_keys was called on the cnn_patterns_detector instance
    mock_cnn_patterns_instance.get_all_detectable_pattern_keys.assert_called_once()


# --- End of new test function ---

# Imports for StrategyManager tests
from core.strategy_manager import StrategyManager
from typing import Dict, Any, Optional, List, Tuple # For type hinting in MockParamsManager


class MockParamsManager:
    def __init__(self):
        self.params: Dict[str, Any] = {}
        self.calls_set_param: List[Tuple[str, Any, str]] = []
        self.calls_update_roi_sl: List[Dict[str, Any]] = []
        self.mock_set_param = AsyncMock(side_effect=self._set_param_impl)
        self.mock_update_strategy_roi_sl_params = AsyncMock(side_effect=self._update_strategy_roi_sl_params_impl)
        self.mock_get_param = AsyncMock(side_effect=self._get_param_impl)

    async def _set_param_impl(self, key: str, value: Any, strategy_id: str):
        self.calls_set_param.append((key, value, strategy_id))
        if strategy_id not in self.params:
            self.params[strategy_id] = {"buy": {}, "sell": {}, "roi": {}, "stoploss": None, "trailing": {}}
        # Simplistic: assume it's a buy param if not roi, stoploss, or trailing related
        # This mock is primarily for tracking calls, not perfect state simulation.
        if "buy" not in self.params[strategy_id]: self.params[strategy_id]["buy"] = {}
        self.params[strategy_id]["buy"][key] = value
        logging.info(f"MockParamsManager: Set {key} to {value} for {strategy_id}")

    async def _update_strategy_roi_sl_params_impl(self, strategy_id: str, new_roi: Optional[Dict] = None,
                                                new_stoploss: Optional[float] = None,
                                                new_trailing_stop: Optional[float] = None,
                                                new_trailing_only_offset_is_reached: Optional[float] = None):
        call_args = {
            "strategy_id": strategy_id, "new_roi": new_roi, "new_stoploss": new_stoploss,
            "new_trailing_stop": new_trailing_stop,
            "new_trailing_only_offset_is_reached": new_trailing_only_offset_is_reached
        }
        self.calls_update_roi_sl.append(call_args)

        if strategy_id not in self.params:
            self.params[strategy_id] = {"buy": {}, "sell": {}, "roi": {}, "stoploss": None, "trailing": {}}

        if new_roi is not None:
            self.params[strategy_id]['roi'] = new_roi
        if new_stoploss is not None:
            self.params[strategy_id]['stoploss'] = new_stoploss
        if new_trailing_stop is not None:
            if 'trailing' not in self.params[strategy_id]: self.params[strategy_id]['trailing'] = {}
            self.params[strategy_id]['trailing']['value'] = new_trailing_stop
        if new_trailing_only_offset_is_reached is not None:
            if 'trailing' not in self.params[strategy_id]: self.params[strategy_id]['trailing'] = {}
            self.params[strategy_id]['trailing']['offset'] = new_trailing_only_offset_is_reached
        logging.info(f"MockParamsManager: Updated ROI/SL/Trailing for {strategy_id}")

    async def _get_param_impl(self, category: str, strategy_id: str = None, param_name: str = None):
        # This is a simplified get_param for basic support if needed by StrategyManager internally.
        # The primary test focus is on set_param and update_strategy_roi_sl_params.
        if strategy_id and strategy_id in self.params:
            if param_name:
                return self.params[strategy_id].get("buy", {}).get(param_name) or \
                       self.params[strategy_id].get("sell", {}).get(param_name)
            return self.params[strategy_id] # Return whole strategy dict if no specific param_name
        return None

    # Expose AsyncMocks for assertion
    def set_param(self, key: str, value: Any, strategy_id: str):
        return self.mock_set_param(key=key, value=value, strategy_id=strategy_id)

    def update_strategy_roi_sl_params(self, strategy_id: str, new_roi: Optional[Dict] = None,
                                      new_stoploss: Optional[float] = None,
                                      new_trailing_stop: Optional[float] = None,
                                      new_trailing_only_offset_is_reached: Optional[float] = None):
        return self.mock_update_strategy_roi_sl_params(
            strategy_id=strategy_id, new_roi=new_roi, new_stoploss=new_stoploss,
            new_trailing_stop=new_trailing_stop,
            new_trailing_only_offset_is_reached=new_trailing_only_offset_is_reached
        )

    def get_param(self, category: str, strategy_id: str = None, param_name: str = None):
        return self.mock_get_param(category=category, strategy_id=strategy_id, param_name=param_name)


class TestStrategyManager:
    @pytest.mark.asyncio
    async def test_mutate_strategy_updates_params_manager_correctly(self):
        mock_params_manager = MockParamsManager()
        # db_path=None is okay because mutate_strategy doesn't use the DB.
        strategy_manager = StrategyManager(db_path=None)
        strategy_manager.params_manager = mock_params_manager # Inject mock

        test_strategy_id = "DUOAI_Strategy"
        mock_proposal = {
            "strategyId": test_strategy_id,
            "adjustments": {
                "parameterChanges": {"emaPeriod": 30, "rsiThreshold": 70},
                "roi": {0: 0.15, 30: 0.10, 90: 0.05},
                "stoploss": -0.12,
                "trailingStop": {"value": 0.03, "offset": 0.006}
            },
            "confidence": 0.95,
            "rationale": "Test mutation proposal"
        }

        result = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal)
        assert result is True

        # Verify set_param calls
        mock_params_manager.mock_set_param.assert_any_call(key="emaPeriod", value=30, strategy_id=test_strategy_id)
        mock_params_manager.mock_set_param.assert_any_call(key="rsiThreshold", value=70, strategy_id=test_strategy_id)
        assert mock_params_manager.mock_set_param.call_count == 2

        # Verify update_strategy_roi_sl_params call
        mock_params_manager.mock_update_strategy_roi_sl_params.assert_called_once_with(
            strategy_id=test_strategy_id,
            new_roi={0: 0.15, 30: 0.10, 90: 0.05},
            new_stoploss=-0.12,
            new_trailing_stop=0.03,
            new_trailing_only_offset_is_reached=0.006
        )

    @pytest.mark.asyncio
    async def test_mutate_strategy_no_changes_returns_false(self):
        mock_params_manager = MockParamsManager()
        strategy_manager = StrategyManager(db_path=None)
        strategy_manager.params_manager = mock_params_manager

        test_strategy_id = "DUOAI_Strategy_NoChange"
        mock_proposal_no_actionable_changes = {
            "strategyId": test_strategy_id,
            "adjustments": {
                # No parameterChanges, roi, stoploss, or trailingStop
            },
            "confidence": 0.90,
            "rationale": "Test no changes proposal"
        }

        result = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal_no_actionable_changes)
        assert result is False
        mock_params_manager.mock_set_param.assert_not_called()
        mock_params_manager.mock_update_strategy_roi_sl_params.assert_not_called()

        # Test with empty adjustments
        mock_proposal_empty_adjustments = {
            "strategyId": test_strategy_id,
            "adjustments": {}, # Empty
            "confidence": 0.90,
            "rationale": "Test empty adjustments"
        }
        result_empty = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal_empty_adjustments)
        assert result_empty is False
        mock_params_manager.mock_set_param.assert_not_called() # Still not called
        mock_params_manager.mock_update_strategy_roi_sl_params.assert_not_called() # Still not called


    @pytest.mark.asyncio
    async def test_mutate_strategy_invalid_proposal_returns_false(self):
        mock_params_manager = MockParamsManager()
        strategy_manager = StrategyManager(db_path=None)
        strategy_manager.params_manager = mock_params_manager

        test_strategy_id = "DUOAI_Strategy_Invalid"

        # Test with None proposal
        result_none = await strategy_manager.mutate_strategy(test_strategy_id, None)
        assert result_none is False

        # Test with empty proposal dict
        result_empty = await strategy_manager.mutate_strategy(test_strategy_id, {})
        assert result_empty is False

        # Test with mismatched strategyId
        mock_proposal_wrong_id = {
            "strategyId": "DifferentStrategy",
            "adjustments": {"parameterChanges": {"emaPeriod": 10}}
        }
        result_wrong_id = await strategy_manager.mutate_strategy(test_strategy_id, mock_proposal_wrong_id)
        assert result_wrong_id is False

        mock_params_manager.mock_set_param.assert_not_called()
        mock_params_manager.mock_update_strategy_roi_sl_params.assert_not_called()


# Helper function to extract JSON blocks from prompt string (adapted from core/prompt_builder.py test)
def extract_json_block_from_prompt(prompt_text: str, start_marker: str, end_marker: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON block from a given text, identified by start and optional end markers.
    """
    try:
        # Ensure start_marker is treated as a literal string for splitting
        split_after_start = prompt_text.split(start_marker, 1)
        if len(split_after_start) < 2:
            # logging.error(f"Start marker '{start_marker}' not found in prompt.")
            return None

        relevant_text_part = split_after_start[1]

        # If an end_marker is provided and found, split by it
        if end_marker:
            split_before_end = relevant_text_part.split(end_marker, 1)
            if len(split_before_end) < 2: # End marker not found after start marker
                # This might be okay if the JSON block is the last part of the prompt
                json_str = relevant_text_part.strip()
            else:
                json_str = split_before_end[0].strip()
        else:
            # Otherwise, assume the JSON string goes to the end of the text (or before next section)
            # This requires careful selection of start_marker or assumes JSON is at the end.
            json_str = relevant_text_part.strip()

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # logging.error(f"JSONDecodeError for block starting with '{start_marker}': {e}")
        # logging.error(f"Problematic JSON string part: {json_str[:300]}...") # Log more context
        return None
    except Exception as e:
        # logging.error(f"An unexpected error occurred in extract_json_block: {e}")
        return None


# Imports for PromptBuilder tests
from core.prompt_builder import PromptBuilder
# pandas (pd) and numpy (np) are already imported. json is also available.

class TestPromptBuilder:
    @pytest.mark.asyncio
    async def test_generate_prompt_with_complete_data(self):
        builder = PromptBuilder()

        # Create mock candles_by_timeframe
        mock_candles = {
            '5m': pd.DataFrame({
                'open': [10, 11, 12, 11.5, 12.55],
                'high': [10.5, 11.5, 12.5, 12, 13.01],
                'low': [9.5, 10.5, 11.5, 11, 12.02],
                'close': [11, 12, 11.5, 12.5, 12.88],
                'volume': [100, 120, 110, 130, 90.5],
                'rsi': [50, 55, 60, 58, 62.12],
                'macd': [0.1, 0.12, 0.15, 0.13, 0.161],
                'macdsignal': [0.09, 0.1, 0.11, 0.12, 0.132],
                'macdhist': [0.01, 0.02, 0.04, 0.01, 0.033],
                'bb_lowerband': [9, 10, 11, 10.5, 11.54],
                'bb_middleband': [10, 11, 12, 11.5, 12.55],
                'bb_upperband': [11, 12, 13, 12.5, 13.56],
                'ema_short': [10.8, 11.2, 11.8, 11.6, 12.27],
                'ema_long': [10.5, 10.8, 11.1, 11.3, 11.78],
                'ema_20': [10.7, 11.1, 11.7, 11.5, 12.19],
                'ema_100': [10.2, 10.5, 10.8, 11.0, 11.41],
                'volume_mean_10': [110, 115, 112, 118, 110.52],
                'volume_mean_30': [105, 110, 115, 112, 108.03],
                'CDLMORNINGSTAR': [0, 0, 0, 0, 100],    # Active pattern
                'CDLEVENINGSTAR': [0, 0, 0, 0, 0],     # Inactive pattern (value 0)
                'CDLDOJI': [0, 0, 100, 0, 0],          # Pattern not on latest candle
                'CDLHAMMER': [0,0,0,0, -100]           # Active negative pattern
            }),
            '1h': pd.DataFrame({
                'open': [1200, 1210, 1205.1],
                'high': [1250, 1220, 1215.2],
                'low': [1190, 1200, 1200.3],
                'close': [1210, 1205, 1212.77],
                'volume': [1000, 1200, 1100.5],
                'rsi': [60.3, 55.2, 58.99],
                'ema_short': [1208, 1206, 1209.51],
                'ema_50': [1200, 1202, 1205.02],
                'CDLENGULFING': [0, 0, -100],          # Active pattern
                'CDLSHOOTINGSTAR': [0,0,0]             # Inactive
            })
        }

        # Create mock additional_context
        mock_additional_context = {
            'pattern_data': {
                'cnn_predictions': {
                    '5m_bullFlag_score': 0.92,
                    '1h_bearTrap_score': 0.78,
                    '5m_randomPattern_score': 0.3 # Should also be included
                }
            },
            'social_sentiment_data': [
                {
                    "title": "TEST: BTC to the moon!", "source": "TestCryptoNews",
                    "content": "Test content about BTC rally.", "timestamp": "2024-01-01T10:00:00Z"
                }
            ]
        }

        prompt_string = await builder.generate_prompt_with_data(
            candles_by_timeframe=mock_candles,
            symbol="TEST/USDT",
            prompt_type="marketAnalysis",
            current_bias=0.5,
            current_confidence=0.6,
            additional_context=mock_additional_context
        )

        assert prompt_string is not None
        assert isinstance(prompt_string, str)

        # Technical Indicators Assertions
        tech_indicators_json = extract_json_block_from_prompt(prompt_string, "Latest Technical Indicators:\n", "\nDetected Candlestick Patterns:")
        assert tech_indicators_json is not None, "Technical indicators JSON block not found or invalid."

        # 5m timeframe indicators
        assert '5m' in tech_indicators_json
        indicators_5m = tech_indicators_json['5m']
        latest_5m_row = mock_candles['5m'].iloc[-1]
        expected_indicators_5m = {
            'rsi': latest_5m_row['rsi'], 'macd': latest_5m_row['macd'],
            'macdsignal': latest_5m_row['macdsignal'], 'macdhist': latest_5m_row['macdhist'],
            'bb_lowerband': latest_5m_row['bb_lowerband'], 'bb_middleband': latest_5m_row['bb_middleband'],
            'bb_upperband': latest_5m_row['bb_upperband'], 'ema_short': latest_5m_row['ema_short'],
            'ema_long': latest_5m_row['ema_long'], 'ema_20': latest_5m_row['ema_20'],
            'ema_100': latest_5m_row['ema_100'], 'volume_mean_10': latest_5m_row['volume_mean_10'],
            'volume_mean_30': latest_5m_row['volume_mean_30']
        }
        for k, v in expected_indicators_5m.items():
            assert k in indicators_5m, f"Indicator {k} missing for 5m"
            assert abs(indicators_5m[k] - v) < 1e-9, f"Indicator {k} value mismatch for 5m"

        # 1h timeframe indicators
        assert '1h' in tech_indicators_json
        indicators_1h = tech_indicators_json['1h']
        latest_1h_row = mock_candles['1h'].iloc[-1]
        expected_indicators_1h = {
            'rsi': latest_1h_row['rsi'],
            'ema_short': latest_1h_row['ema_short'], 'ema_50': latest_1h_row['ema_50']
        }
        for k, v in expected_indicators_1h.items():
            assert k in indicators_1h, f"Indicator {k} missing for 1h"
            assert abs(indicators_1h[k] - v) < 1e-9, f"Indicator {k} value mismatch for 1h"
        assert 'macd' not in indicators_1h # Check that not all indicators are present if not in source df

        # Candlestick Patterns Assertions
        candlestick_json = extract_json_block_from_prompt(prompt_string, "Detected Candlestick Patterns:\n", "\nDetected CNN Patterns:")
        assert candlestick_json is not None, "Candlestick patterns JSON block not found or invalid."

        assert '5m' in candlestick_json
        assert "CDLMORNINGSTAR" in candlestick_json['5m']
        assert "CDLHAMMER" in candlestick_json['5m']
        assert "CDLEVENINGSTAR" not in candlestick_json['5m'] # Value was 0
        assert "CDLDOJI" not in candlestick_json['5m'] # Not on latest row

        assert '1h' in candlestick_json
        assert "CDLENGULFING" in candlestick_json['1h']
        assert "CDLSHOOTINGSTAR" not in candlestick_json['1h'] # Value was 0

        # CNN Patterns Assertions
        cnn_patterns_json = extract_json_block_from_prompt(prompt_string, "Detected CNN Patterns:\n", "\nLatest Social Sentiment/News:")
        assert cnn_patterns_json is not None, "CNN patterns JSON block not found or invalid."
        expected_cnn_predictions = mock_additional_context['pattern_data']['cnn_predictions']
        for k, v in expected_cnn_predictions.items():
            assert k in cnn_patterns_json
            assert cnn_patterns_json[k] == v

        # Basic OHLCV Data Presence
        assert "Latest 5 candles (condensed):" in prompt_string
        assert "12.88" in prompt_string # From 5m close
        assert "1212.77" in prompt_string # From 1h close

        # Social Sentiment Data Presence
        assert "Latest Social Sentiment/News:" in prompt_string
        assert "Title: TEST: BTC to the moon!" in prompt_string
        assert "Source: TestCryptoNews" in prompt_string
        assert "Snippet: Test content about BTC rally." in prompt_string # Content is short, so full content
        assert "Timestamp: 2024-01-01T10:00:00Z" in prompt_string


# Imports for ConfidenceEngine tests
from core.confidence_engine import ConfidenceEngine, CONFIDENCE_COOLDOWN_SECONDS
# datetime, timedelta, os, json, asyncio, patch, logging are already imported or available via pytest.

@pytest.fixture
def confidence_engine_instance_with_temp_memory(tmp_path):
    """
    Provides a ConfidenceEngine instance using a temporary CONFIDENCE_MEMORY_FILE.
    Ensures test isolation by providing a clean memory file for each test.
    """
    temp_memory_dir = tmp_path / "confidence_memory_test"
    temp_memory_dir.mkdir()
    temp_confidence_file_path = temp_memory_dir / "confidence_memory.json"

    with patch('core.confidence_engine.CONFIDENCE_MEMORY_FILE', str(temp_confidence_file_path)):
        engine = ConfidenceEngine()
        yield engine, str(temp_confidence_file_path)
    # Cleanup handled by tmp_path

class TestConfidenceEngine:
    TEST_TOKEN = "TEST_CONF/TOKEN"
    TEST_STRATEGY = "TestConfStrategy"

    @pytest.mark.asyncio
    async def test_initial_confidence_and_max_pct(self, confidence_engine_instance_with_temp_memory, caplog):
        """Checks default confidence and max_per_trade_pct for a new token/strategy."""
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory

        initial_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        initial_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert initial_confidence == 0.5, "Initial confidence should be 0.5."
        assert initial_max_pct == 0.1, "Initial max_per_trade_pct should be 0.1."
        assert f"ConfidenceEngine geïnitialiseerd. Geheugen: {engine.confidence_memory}" in caplog.text # Check init log

    @pytest.mark.asyncio
    async def test_update_confidence_winning_trade(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, memory_file = confidence_engine_instance_with_temp_memory

        initial_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY) # 0.5
        initial_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY) # 0.1

        ai_reported_confidence = 0.8
        trade_profit_pct = 0.02 # 2% profit

        await engine.update_confidence(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            ai_reported_confidence=ai_reported_confidence,
            trade_result_pct=trade_profit_pct
        )

        updated_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        updated_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Expected calculations:
        # profit_boost = min(0.02 * 5, 1.0) = 0.1
        # confidence_increase = (0.8 * 0.05) + (0.1 * 0.1) = 0.04 + 0.01 = 0.05
        # adjusted_learned_confidence = 0.5 + 0.05 = 0.55
        # max_per_trade_pct = min(0.1 + 0.01 + (0.1 * 0.02), 0.5) = min(0.11 + 0.002, 0.5) = min(0.112, 0.5) = 0.112

        assert abs(updated_confidence - 0.55) < 0.0001, "Confidence calculation mismatch for winning trade."
        assert abs(updated_max_pct - 0.112) < 0.0001, "MaxPerTradePct calculation mismatch for winning trade."
        assert f"Confidence voor {self.TEST_TOKEN}/{self.TEST_STRATEGY} bijgewerkt naar {updated_confidence:.3f}" in caplog.text

        with open(memory_file, 'r') as f:
            memory = json.load(f)
        assert memory[self.TEST_TOKEN][self.TEST_STRATEGY]['confidence'] == updated_confidence
        assert memory[self.TEST_TOKEN][self.TEST_STRATEGY]['max_per_trade_pct'] == updated_max_pct
        assert memory[self.TEST_TOKEN][self.TEST_STRATEGY]['trade_count'] == 1

    @pytest.mark.asyncio
    async def test_update_confidence_large_winning_trade(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory
        # Assume previous state or start fresh; for this test, starting fresh is fine as logic is additive

        ai_reported_confidence = 0.9
        trade_profit_pct = 0.05 # 5% profit

        await engine.update_confidence(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            ai_reported_confidence=ai_reported_confidence,
            trade_result_pct=trade_profit_pct
        )
        updated_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        updated_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # profit_boost = min(0.05 * 5, 1.0) = 0.25
        # confidence_increase = (0.9 * 0.05) + (0.25 * 0.1) = 0.045 + 0.025 = 0.07
        # adjusted_learned_confidence = 0.5 (initial) + 0.07 = 0.57
        # max_per_trade_pct = min(0.1 + 0.01 + (0.25 * 0.02), 0.5) = min(0.11 + 0.005, 0.5) = min(0.115, 0.5) = 0.115
        assert abs(updated_confidence - 0.57) < 0.0001
        assert abs(updated_max_pct - 0.115) < 0.0001

    @pytest.mark.asyncio
    async def test_update_confidence_losing_trade(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory

        ai_reported_confidence = 0.7
        trade_profit_pct = -0.01 # -1% loss

        await engine.update_confidence(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            ai_reported_confidence=ai_reported_confidence,
            trade_result_pct=trade_profit_pct
        )
        updated_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        updated_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # loss_penalty = min(abs(-0.01) * 5, 1.0) = 0.05
        # confidence_decrease = (0.7 * 0.05) + (0.05 * 0.1) = 0.035 + 0.005 = 0.04
        # adjusted_learned_confidence = 0.5 - 0.04 = 0.46
        # max_per_trade_pct = max(0.1 - 0.01 - (0.05 * 0.02), 0.02) = max(0.09 - 0.001, 0.02) = max(0.089, 0.02) = 0.089
        assert abs(updated_confidence - 0.46) < 0.0001
        assert abs(updated_max_pct - 0.089) < 0.0001

    @pytest.mark.asyncio
    async def test_update_confidence_large_losing_trade(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory

        ai_reported_confidence = 0.8
        trade_profit_pct = -0.04 # -4% loss

        await engine.update_confidence(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            ai_reported_confidence=ai_reported_confidence,
            trade_result_pct=trade_profit_pct
        )
        updated_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        updated_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # loss_penalty = min(abs(-0.04) * 5, 1.0) = 0.20
        # confidence_decrease = (0.8 * 0.05) + (0.20 * 0.1) = 0.04 + 0.02 = 0.06
        # adjusted_learned_confidence = 0.5 - 0.06 = 0.44
        # max_per_trade_pct = max(0.1 - 0.01 - (0.20 * 0.02), 0.02) = max(0.09 - 0.004, 0.02) = max(0.086, 0.02) = 0.086
        assert abs(updated_confidence - 0.44) < 0.0001
        assert abs(updated_max_pct - 0.086) < 0.0001

    @pytest.mark.asyncio
    async def test_update_confidence_ai_only_reflection(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory
        initial_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY) # 0.5
        initial_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY) # 0.1

        ai_reported_confidence = 0.9
        await engine.update_confidence(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            ai_reported_confidence=ai_reported_confidence
            # No trade_result_pct
        )
        updated_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        updated_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # adjustment_weight = 0.1
        # expected_confidence = (0.5 * (1 - 0.1) + 0.9 * 0.1) = (0.5 * 0.9 + 0.09) = 0.45 + 0.09 = 0.54
        assert abs(updated_confidence - 0.54) < 0.0001
        assert updated_max_pct == initial_max_pct, "MaxPerTradePct should not change on AI-only reflection."

    @pytest.mark.asyncio
    async def test_confidence_update_cooldown(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.DEBUG)
        engine, _ = confidence_engine_instance_with_temp_memory

        # First update
        await engine.update_confidence(self.TEST_TOKEN, self.TEST_STRATEGY, 0.7, trade_result_pct=0.01)
        first_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        first_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Attempt second update immediately
        await engine.update_confidence(self.TEST_TOKEN, self.TEST_STRATEGY, 0.2, trade_result_pct=-0.02)
        second_confidence = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        second_max_pct = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert abs(second_confidence - first_confidence) < 0.0001, "Confidence should not change during cooldown."
        assert abs(second_max_pct - first_max_pct) < 0.0001, "MaxPerTradePct should not change during cooldown."
        assert f"Cooldown actief voor {self.TEST_TOKEN}/{self.TEST_STRATEGY}" in caplog.text

    @pytest.mark.asyncio
    async def test_confidence_update_after_cooldown_mocked_time(self, confidence_engine_instance_with_temp_memory, caplog):
        caplog.set_level(logging.INFO)
        engine, _ = confidence_engine_instance_with_temp_memory

        initial_update_time = datetime(2023, 1, 1, 12, 0, 0)

        # Mock datetime for the first update
        with patch('core.confidence_engine.datetime') as mock_dt_initial:
            mock_dt_initial.now.return_value = initial_update_time
            mock_dt_initial.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)
            await engine.update_confidence(self.TEST_TOKEN, self.TEST_STRATEGY, 0.75, trade_result_pct=0.01) # win

        confidence_after_first_update = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        max_pct_after_first_update = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Mock datetime for an attempt during cooldown
        with patch('core.confidence_engine.datetime') as mock_dt_cooldown:
            mock_dt_cooldown.now.return_value = initial_update_time + timedelta(seconds=CONFIDENCE_COOLDOWN_SECONDS / 2)
            mock_dt_cooldown.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)
            await engine.update_confidence(self.TEST_TOKEN, self.TEST_STRATEGY, 0.3, trade_result_pct=-0.01) # loss

        assert abs(engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY) - confidence_after_first_update) < 0.0001
        assert abs(engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY) - max_pct_after_first_update) < 0.0001

        # Mock datetime for an attempt after cooldown
        with patch('core.confidence_engine.datetime') as mock_dt_after_cooldown:
            mock_dt_after_cooldown.now.return_value = initial_update_time + timedelta(seconds=CONFIDENCE_COOLDOWN_SECONDS + 1)
            mock_dt_after_cooldown.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)
            await engine.update_confidence(self.TEST_TOKEN, self.TEST_STRATEGY, 0.3, trade_result_pct=-0.01) # loss

        confidence_after_cooldown_expiry = engine.get_confidence_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        max_pct_after_cooldown_expiry = engine.get_max_per_trade_pct(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert confidence_after_cooldown_expiry != confidence_after_first_update, "Confidence should have updated after cooldown."
        assert max_pct_after_cooldown_expiry != max_pct_after_first_update, "MaxPerTradePct should have updated after cooldown."




# Imports for CNNPatternsMain tests
from core.cnn_patterns import CNNPatterns, SimpleCNN # SimpleCNN needed for creating dummy model
# torch, pd, np, os, json, datetime, logging, asyncio, sys, pytest, patch, MagicMock already imported or available

@pytest.fixture
def cnn_patterns_with_dummy_models(tmp_path):
    """
    Provides a CNNPatterns instance with dummy models and scalers in a temporary directory.
    Patches CNNPatterns.MODELS_DIR and core.params_manager.ParamsManager for the test.
    """
    dummy_models_dir = tmp_path / "data" / "models"

    test_symbol_underscore = "TEST_USDT"
    test_timeframe = "1h"
    test_arch_key = "default_simple"

    # Mock ParamsManager for CNNPatterns instance
    mock_pm_instance = MagicMock(spec=ParamsManager)
    mock_pm_instance.get_param.side_effect = lambda key, default=None: \
        test_arch_key if key == 'current_cnn_architecture_key' else \
        {
            test_arch_key: { # Must match parameters SimpleCNN expects or can handle with defaults
                "num_conv_layers": 2, "filters_per_layer": [16, 32],
                "kernel_sizes_per_layer": [3, 3], "strides_per_layer": [1, 1],
                "padding_per_layer": [1, 1], "pooling_types_per_layer": ['max', 'max'],
                "pooling_kernel_sizes_per_layer": [2, 2], "pooling_strides_per_layer": [2, 2],
                "use_batch_norm": False, "dropout_rate": 0.0
            }
        } if key == 'cnn_architecture_configs' else \
        default

    # Create dummy model and scaler files
    dummy_patterns_meta = {
        'bullFlag': {'input_channels': 12, 'num_classes': 2},
        'bearishEngulfing': {'input_channels': 12, 'num_classes': 2}
    }
    dummy_sequence_length = 30
    dummy_feature_names = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']


    for pattern_name, meta_info in dummy_patterns_meta.items():
        model_dir_path = dummy_models_dir / test_symbol_underscore / test_timeframe
        model_dir_path.mkdir(parents=True, exist_ok=True)

        model_path = model_dir_path / f"cnn_model_{pattern_name}_{test_arch_key}.pth"
        scaler_path = model_dir_path / f"scaler_params_{pattern_name}_{test_arch_key}.json"

        # Create dummy model state_dict
        # Ensure input_channels in SimpleCNN matches len(dummy_feature_names)
        arch_params_for_dummy_model = mock_pm_instance.get_param('cnn_architecture_configs')[test_arch_key]
        dummy_model = SimpleCNN(
            input_channels=len(dummy_feature_names),
            num_classes=meta_info['num_classes'],
            sequence_length=dummy_sequence_length,
            **arch_params_for_dummy_model
            )
        torch.save(dummy_model.state_dict(), model_path)

        # Create dummy scaler JSON
        dummy_scaler_params = {
            'min_': [0.0] * len(dummy_feature_names),
            'scale_': [1.0] * len(dummy_feature_names),
            'sequence_length': dummy_sequence_length,
            'feature_names_in_': dummy_feature_names
        }
        with open(scaler_path, 'w') as f:
            json.dump(dummy_scaler_params, f)

    with patch('core.cnn_patterns.CNNPatterns.MODELS_DIR', str(dummy_models_dir)), \
         patch('core.cnn_patterns.ParamsManager', return_value=mock_pm_instance):
        cnn_detector = CNNPatterns()
        # Pre-load models for the test symbol/timeframe to ensure they are loaded from dummy files
        cnn_detector._load_cnn_models_and_scalers(symbol=test_symbol_underscore.replace("_", "/"), timeframe=test_timeframe, arch_key_to_load=test_arch_key)
        yield cnn_detector, test_symbol_underscore.replace("_", "/"), test_timeframe # Return symbol and timeframe for use in test

class TestCNNPatternsMain:
    def _create_mock_dataframe(self, num_rows=50):
        return pd.DataFrame({
            'open': np.random.rand(num_rows) * 100, 'high': np.random.rand(num_rows) * 110,
            'low': np.random.rand(num_rows) * 90, 'close': np.random.rand(num_rows) * 100,
            'volume': np.random.rand(num_rows) * 1000, 'rsi': np.random.rand(num_rows) * 100,
            'macd': np.random.rand(num_rows), 'macdsignal': np.random.rand(num_rows), 'macdhist': np.random.rand(num_rows),
            'bb_lowerband': np.random.rand(num_rows)*95, 'bb_middleband':np.random.rand(num_rows)*100, 'bb_upperband':np.random.rand(num_rows)*105,
            'date': pd.date_range(end=datetime.now(timezone.utc), periods=num_rows, freq='1h')
        }).set_index('date')

    @pytest.mark.asyncio
    async def test_cnn_patterns_main_loading_and_prediction(self, cnn_patterns_with_dummy_models, caplog):
        caplog.set_level(logging.DEBUG)
        cnn_detector, test_symbol, test_timeframe = cnn_patterns_with_dummy_models

        mock_df = self._create_mock_dataframe()

        # Test predict_pattern_score for patterns with dummy models
        for pattern_name in cnn_detector.pattern_model_meta_info.keys():
            score = await cnn_detector.predict_pattern_score(mock_df, test_symbol, test_timeframe, pattern_name)
            assert 0.0 <= score <= 1.0, f"{pattern_name} score out of range: {score}"
            assert f"CNN voorspelde score voor '{test_symbol.replace('/', '_')}_{test_timeframe}_{pattern_name}_default_simple'" in caplog.text

    @pytest.mark.asyncio
    async def test_cnn_patterns_main_get_all_detectable_keys(self, cnn_patterns_with_dummy_models):
        cnn_detector, _, _ = cnn_patterns_with_dummy_models
        pattern_keys = cnn_detector.get_all_detectable_pattern_keys()
        expected_keys = list(cnn_detector.pattern_model_meta_info.keys())
        assert sorted(pattern_keys) == sorted(expected_keys), \
            f"Pattern keys {pattern_keys} do not match expected {expected_keys}"

    @pytest.mark.asyncio
    async def test_cnn_patterns_main_determine_trend(self, cnn_patterns_with_dummy_models, caplog):
        caplog.set_level(logging.DEBUG)
        cnn_detector, _, _ = cnn_patterns_with_dummy_models # Need instance for determine_trend
        base_time_dt = datetime.now(timezone.utc)

        # Helper to create list of candle dicts
        def create_candle_list(prices):
            return [{'open': p, 'high': p + 1, 'low': p - 1, 'close': p, 'volume': 100,
                       'time': (base_time_dt + timedelta(hours=i)).timestamp() * 1000}
                      for i, p in enumerate(prices)]

        # Uptrend
        uptrend_prices = [50 + i * 0.5 for i in range(20)] # Clearer trend
        trend_result = cnn_detector.determine_trend(create_candle_list(uptrend_prices))
        assert trend_result == "uptrend", f"Expected 'uptrend', got {trend_result}"

        # Downtrend
        downtrend_prices = [70 - i * 0.5 for i in range(20)] # Clearer trend
        trend_result = cnn_detector.determine_trend(create_candle_list(downtrend_prices))
        assert trend_result == "downtrend", f"Expected 'downtrend', got {trend_result}"

        # Sideways Trend (very little change)
        sideways_prices = [60 + ( (i % 3 - 1) * 0.01 ) for i in range(20)]
        trend_result = cnn_detector.determine_trend(create_candle_list(sideways_prices))
        assert trend_result == "sideways", f"Expected 'sideways', got {trend_result}"

        # Insufficient Data
        insufficient_candles = create_candle_list(uptrend_prices[:5])
        trend_result = cnn_detector.determine_trend(insufficient_candles)
        assert trend_result == "undetermined", f"Expected 'undetermined' for insufficient data, got {trend_result}"

    @pytest.mark.asyncio
    async def test_cnn_patterns_main_detect_volume_spike(self, cnn_patterns_with_dummy_models, caplog):
        caplog.set_level(logging.DEBUG)
        cnn_detector, _, _ = cnn_patterns_with_dummy_models
        base_time_dt = datetime.now(timezone.utc)

        def create_candle_list_vol(volumes):
            return [{'open': 100, 'high': 105, 'low': 95, 'close': 100, 'volume': v,
                       'time': (base_time_dt + timedelta(hours=i)).timestamp() * 1000}
                      for i, v in enumerate(volumes)]

        # Volume Spike
        spike_volumes = [100] * 19 + [300] # Last volume is 3x average
        spike_result = cnn_detector.detect_volume_spike(create_candle_list_vol(spike_volumes), period=20, spike_factor=2.0)
        assert spike_result is True, "Expected True for volume spike"

        # No Volume Spike
        no_spike_volumes = [100] * 20
        no_spike_volumes[-1] = 110 # Last volume slightly higher, but not a spike
        spike_result = cnn_detector.detect_volume_spike(create_candle_list_vol(no_spike_volumes), period=20, spike_factor=2.0)
        assert spike_result is False, "Expected False for no volume spike"

        # Insufficient Data
        insufficient_volumes = spike_volumes[:5]
        spike_result = cnn_detector.detect_volume_spike(create_candle_list_vol(insufficient_volumes), period=20, spike_factor=2.0)
        assert spike_result is False, "Expected False for insufficient data (volume spike)"

    @pytest.mark.asyncio
    async def test_detect_patterns_multi_timeframe_output_structure(self, cnn_patterns_with_dummy_models, caplog):
        """
        Tests the output structure of detect_patterns_multi_timeframe,
        focusing on cnn_predictions and rule-based pattern integration.
        """
        caplog.set_level(logging.DEBUG)
        cnn_detector, test_symbol, test_timeframe_from_fixture = cnn_patterns_with_dummy_models

        # 1. Mock predict_pattern_score
        async def mock_predict_score(df, symbol, timeframe, pattern_name):
            if symbol == test_symbol:
                if pattern_name == 'bullFlag' and timeframe == '5m': return 0.8
                if pattern_name == 'bearishEngulfing' and timeframe == '1h': return 0.7
                if pattern_name == 'bullFlag' and timeframe == '15m': return 0.0 # Score that should be ignored or recorded as 0
            return 0.0 # Default score for other cases
        cnn_detector.predict_pattern_score = AsyncMock(side_effect=mock_predict_score)

        # 2. Mock rule-based detection methods (simplified)
        # These are methods of CNNPatterns instance.
        cnn_detector.detect_bull_flag = MagicMock(return_value=True) # Assume it's a method taking candles list
        cnn_detector._detect_engulfing = MagicMock(return_value="bearishEngulfing") # Takes candles list
        # detect_candlestick_patterns takes a DataFrame
        cnn_detector.detect_candlestick_patterns = MagicMock(return_value={"CDLDOJI": True, "CDLHAMMER": False}) # Example output

        # Mock context methods
        cnn_detector.determine_trend = MagicMock(return_value="uptrend")
        cnn_detector.detect_volume_spike = MagicMock(return_value=True)


        # 3. Prepare mock DataFrames for different timeframes
        mock_df_5m = self._create_mock_dataframe(num_rows=30, timeframe='5m')
        mock_df_15m = self._create_mock_dataframe(num_rows=30, timeframe='15m')
        mock_df_1h = self._create_mock_dataframe(num_rows=30, timeframe='1h')
        candles_by_tf = {'5m': mock_df_5m, '15m': mock_df_15m, '1h': mock_df_1h}

        # 4. Call detect_patterns_multi_timeframe
        output = await cnn_detector.detect_patterns_multi_timeframe(candles_by_tf, test_symbol)

        # 5. Assertions for cnn_predictions
        assert "cnn_predictions" in output
        cnn_preds = output["cnn_predictions"]
        assert "5m_bullFlag_score" in cnn_preds
        assert cnn_preds["5m_bullFlag_score"] == 0.8
        assert "1h_bearishEngulfing_score" in cnn_preds
        assert cnn_preds["1h_bearishEngulfing_score"] == 0.7
        # Pattern with 0.0 score might be absent or present as 0.0 depending on implementation.
        # Current EntryDecider logic: `if score is not None and isinstance(score, (float, int)) and score > 0:`
        # CNNPatterns log: `if score > 0.0: all_patterns["cnn_predictions"][...] = score`
        # So, 0.0 scores should not be in the output cnn_predictions.
        assert "15m_bullFlag_score" not in cnn_preds, "Scores of 0.0 should not be included in cnn_predictions"

        # 6. Assertions for rule-based patterns in "patterns"
        # The consolidation logic in detect_patterns_multi_timeframe is:
        # `for tf_type_key in ["zoomPatterns", "contextPatterns"]:`
        # `  for patterns_on_tf in all_patterns[tf_type_key].values():`
        # `    for pattern_name, status in patterns_on_tf.items():`
        # `      if status: all_patterns["patterns"][pattern_name] = status`
        # This means the "patterns" dict will have a flat list of detected rule patterns.
        assert "patterns" in output
        rule_patterns = output["patterns"]
        assert rule_patterns.get("bullFlag") is True # From mocked detect_bull_flag
        assert rule_patterns.get("bearishEngulfing") is True # From mocked _detect_engulfing
        assert rule_patterns.get("CDLDOJI") is True # From mocked detect_candlestick_patterns
        assert "CDLHAMMER" not in rule_patterns # Since its mock value was False

        # 7. Assertions for context
        assert "context" in output
        assert output["context"]["trend"] == "uptrend"
        assert output["context"]["volume_spike"] is True

        # Verify mocks were called for different timeframes (example for predict_pattern_score)
        # Calls are (df, symbol, timeframe, pattern_name)
        # Check a few specific calls to ensure iteration over timeframes and patterns
        expected_calls = [
            call(mock_df_5m, test_symbol, '5m', 'bullFlag'),
            call(mock_df_5m, test_symbol, '5m', 'bearishEngulfing'), # Will return 0.0 from mock
            call(mock_df_1h, test_symbol, '1h', 'bullFlag'),       # Will return 0.0 from mock
            call(mock_df_1h, test_symbol, '1h', 'bearishEngulfing'),
            call(mock_df_15m, test_symbol, '15m', 'bullFlag'),
        ]
        cnn_detector.predict_pattern_score.assert_has_calls(expected_calls, any_order=True)
        # Check that rule-based detectors were called for each relevant timeframe
        # Example: cnn_detector.detect_bull_flag would be called with candles_list derived from mock_df_5m, mock_df_15m, mock_df_1h
        # For simplicity, check call count if logic is complex, or specific calls if simpler.
        # Total timeframes with data = 3. `detect_bull_flag` is called per timeframe.
        assert cnn_detector.detect_bull_flag.call_count >= 3 # Called for 5m, 15m, 1h from TIME_FRAME_CONFIG
        assert cnn_detector.detect_candlestick_patterns.call_count >= 3


    @pytest.mark.asyncio
    async def test_dataframe_to_cnn_input(self, cnn_patterns_with_dummy_models, caplog):
        """Tests the _dataframe_to_cnn_input helper method."""
        caplog.set_level(logging.DEBUG)
        cnn_detector, test_symbol, test_timeframe = cnn_patterns_with_dummy_models
        pattern_name = 'bullFlag' # Choose one pattern for which dummy model/scaler exists
        arch_key = "default_simple" # Matches fixture
        model_key = f"{test_symbol.replace('/', '_')}_{test_timeframe}_{pattern_name}_{arch_key}"

        # Ensure the scaler is loaded for this model_key via the fixture's _load_cnn_models_and_scalers call
        assert model_key in cnn_detector.scalers, f"Scaler for {model_key} not loaded by fixture."
        scaler = cnn_detector.scalers[model_key]
        # The dummy scaler expects 12 features as per fixture:
        # ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_lowerband', 'bb_middleband', 'bb_upperband']
        # Sequence length is 30.
        sequence_length = scaler.sequence_length_ # Assuming scaler has sequence_length attribute from dummy json
        num_features = len(scaler.feature_names_in_)


        # 1. Normal case
        normal_df = self._create_mock_dataframe(num_rows=sequence_length + 5, timeframe=test_timeframe)
        # Ensure all necessary feature columns are present in normal_df
        for feature in scaler.feature_names_in_:
            if feature not in normal_df.columns:
                normal_df[feature] = np.random.rand(len(normal_df)) * 10 # Add missing feature if any

        tensor_output = cnn_detector._dataframe_to_cnn_input(normal_df, model_key, sequence_length)
        assert tensor_output is not None
        assert isinstance(tensor_output, torch.Tensor)
        assert tensor_output.shape == (1, num_features, sequence_length) # Batch, Channels (Features), SeqLen
        # Check if data is scaled (approx 0-1). This is a loose check.
        # A more precise check would require knowing exact scaler params and input data.
        assert tensor_output.min() >= -0.1 and tensor_output.max() <= 1.1 # Scaled data should be roughly in [0,1]

        # 2. DataFrame too short
        short_df = self._create_mock_dataframe(num_rows=sequence_length - 1, timeframe=test_timeframe)
        tensor_output_short = cnn_detector._dataframe_to_cnn_input(short_df, model_key, sequence_length)
        assert tensor_output_short is None
        assert f"Niet genoeg candles ({len(short_df)}) voor CNN input (nodig: {sequence_length}) voor model_key '{model_key}'" in caplog.text

        # 3. DataFrame with missing feature columns
        caplog.clear()
        missing_cols_df = self._create_mock_dataframe(num_rows=sequence_length + 5, timeframe=test_timeframe)
        # Remove a column expected by the scaler
        removed_col = scaler.feature_names_in_[0]
        missing_cols_df_dropped = missing_cols_df.drop(columns=[removed_col])
        # _dataframe_to_cnn_input should add it back as NaN, then if NaNs are present, return None.
        # The current _dataframe_to_cnn_input adds missing columns with np.nan, then checks for isnull().values.any().
        tensor_output_missing_cols = cnn_detector._dataframe_to_cnn_input(missing_cols_df_dropped, model_key, sequence_length)
        assert tensor_output_missing_cols is None
        assert f"Kolom '{removed_col}' niet gevonden in dataframe voor model_key '{model_key}'" in caplog.text # Logged as DEBUG
        assert f"NaN waarden gevonden in input data voor CNN voor model_key '{model_key}'" in caplog.text # Logged as WARNING

        # 4. DataFrame with NaN values in critical columns
        caplog.clear()
        nan_df = self._create_mock_dataframe(num_rows=sequence_length + 5, timeframe=test_timeframe)
        for feature in scaler.feature_names_in_: # Ensure all features exist first
            if feature not in nan_df.columns: nan_df[feature] = np.random.rand(len(nan_df))
        nan_df.loc[nan_df.index[-sequence_length//2], scaler.feature_names_in_[0]] = np.nan # Introduce NaN
        tensor_output_nan = cnn_detector._dataframe_to_cnn_input(nan_df, model_key, sequence_length)
        assert tensor_output_nan is None
        assert f"NaN waarden gevonden in input data voor CNN voor model_key '{model_key}'" in caplog.text


# Imports for BitvavoExecutor tests
from core.bitvavo_executor import BitvavoExecutor
import ccxt # For patching and for ccxt error types
from ccxt.base.errors import AuthenticationError, RateLimitExceeded, InsufficientFunds, InvalidOrder, NetworkError, ExchangeError, BadSymbol, RequestTimeout, ExchangeNotAvailable, DDoSProtection

@pytest.fixture
def mock_bitvavo_exchange_fixture():
    """
    Patches ccxt.bitvavo to return an AsyncMock instance representing the exchange.
    This mock_exchange will have its methods (fetch_balance, fetch_ohlcv, etc.)
    as AsyncMocks automatically.
    """
    with patch('ccxt.bitvavo', new_callable=AsyncMock) as mock_ccxt_bitvavo:
        # Configure the instance returned by ccxt.bitvavo() to also be an AsyncMock
        # This represents the 'self.exchange' object in BitvavoExecutor
        mock_exchange_instance = AsyncMock()
        mock_ccxt_bitvavo.return_value = mock_exchange_instance
        yield mock_exchange_instance # Yield the instance that executor.exchange will become

class TestBitvavoExecutor:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"BITVAVO_API_KEY": "test_key", "BITVAVO_SECRET_KEY": "test_secret"})
    async def test_fetch_balance_success(self, mock_bitvavo_exchange_fixture, caplog):
        """Tests successful balance fetching."""
        caplog.set_level(logging.INFO)
        mock_exchange = mock_bitvavo_exchange_fixture

        expected_balance_all = {
            'EUR': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
            'ETH': {'free': 10.0, 'used': 0.0, 'total': 10.0},
            'info': {'some_extra_info': 'value'} # CCXT often includes an 'info' field
        }
        mock_exchange.fetch_balance = AsyncMock(return_value=expected_balance_all)

        executor = BitvavoExecutor() # Initializes with mocked ccxt.bitvavo

        # Test fetching all balances
        balance_all = await executor.fetch_balance()
        assert balance_all == expected_balance_all
        mock_exchange.fetch_balance.assert_called_once()

        # Test fetching specific currency
        mock_exchange.fetch_balance.reset_mock() # Reset for the next call
        expected_balance_eth = expected_balance_all['ETH']
        # No need to reconfigure return_value if the overall structure is what we test against for specific key

        balance_eth = await executor.fetch_balance('ETH')
        assert balance_eth == expected_balance_eth
        mock_exchange.fetch_balance.assert_called_once() # Called again

        # Test fetching non-existent currency
        mock_exchange.fetch_balance.reset_mock()
        balance_non_existent = await executor.fetch_balance('XYZ')
        assert balance_non_existent == {'free': 0, 'used': 0, 'total': 0}
        mock_exchange.fetch_balance.assert_called_once()


    @pytest.mark.asyncio
    @patch.dict(os.environ, {"BITVAVO_API_KEY": "test_key", "BITVAVO_SECRET_KEY": "test_secret"})
    async def test_fetch_ohlcv_success(self, mock_bitvavo_exchange_fixture, caplog):
        """Tests successful OHLCV fetching."""
        caplog.set_level(logging.DEBUG)
        mock_exchange = mock_bitvavo_exchange_fixture

        expected_ohlcv_data = [
            [1672531200000, 100.0, 105.0, 98.0, 102.0, 1000.0], # [timestamp, open, high, low, close, volume]
            [1672534800000, 102.0, 108.0, 101.0, 107.0, 1200.0]
        ]
        mock_exchange.fetch_ohlcv = AsyncMock(return_value=expected_ohlcv_data)

        executor = BitvavoExecutor()
        symbol = "ETH/EUR"
        timeframe = "1h"
        limit = 2

        ohlcv_data = await executor.fetch_ohlcv(symbol, timeframe, limit)

        assert ohlcv_data == expected_ohlcv_data
        mock_exchange.fetch_ohlcv.assert_called_once_with(symbol, timeframe, limit=limit)
        assert f"OHLCV data fetched for {symbol}" in caplog.text


    @pytest.mark.asyncio
    @patch.dict(os.environ, {"BITVAVO_API_KEY": "test_key", "BITVAVO_SECRET_KEY": "test_secret"})
    async def test_create_market_buy_order_success(self, mock_bitvavo_exchange_fixture, caplog):
        caplog.set_level(logging.INFO)
        mock_exchange = mock_bitvavo_exchange_fixture

        expected_order_info = {'id': '12345', 'symbol': 'ETH/EUR', 'status': 'closed', 'amount': 0.01, 'filled': 0.01}
        mock_exchange.create_market_buy_order = AsyncMock(return_value=expected_order_info)

        executor = BitvavoExecutor()
        symbol = "ETH/EUR"
        amount = 0.01

        order_info = await executor.create_market_buy_order(symbol, amount)

        assert order_info == expected_order_info
        mock_exchange.create_market_buy_order.assert_called_once_with(symbol, amount)
        assert f"Market BUY order placed for {amount} of {symbol}" in caplog.text

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"BITVAVO_API_KEY": "test_key", "BITVAVO_SECRET_KEY": "test_secret"})
    async def test_create_market_sell_order_success(self, mock_bitvavo_exchange_fixture, caplog):
        caplog.set_level(logging.INFO)
        mock_exchange = mock_bitvavo_exchange_fixture

        expected_order_info = {'id': '67890', 'symbol': 'BTC/EUR', 'status': 'closed', 'amount': 0.005, 'filled': 0.005}
        mock_exchange.create_market_sell_order = AsyncMock(return_value=expected_order_info)

        executor = BitvavoExecutor()
        symbol = "BTC/EUR"
        amount = 0.005

        order_info = await executor.create_market_sell_order(symbol, amount)

        assert order_info == expected_order_info
        mock_exchange.create_market_sell_order.assert_called_once_with(symbol, amount)
        assert f"Market SELL order placed for {amount} of {symbol}" in caplog.text

    # Parameterize error handling tests
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"BITVAVO_API_KEY": "test_key", "BITVAVO_SECRET_KEY": "test_secret"})
    @pytest.mark.parametrize(
        "method_name, method_args, ccxt_error, expected_return, log_message_part",
        [
            ("fetch_balance", (), AuthenticationError, {}, "AuthenticationError in fetch_balance"),
            ("fetch_balance", ("EUR",), RateLimitExceeded, {}, "RateLimitExceeded in fetch_balance"),
            ("fetch_ohlcv", ("ETH/EUR", "1h", 10), NetworkError, [], "NetworkError in fetch_ohlcv"),
            ("fetch_ohlcv", ("ETH/EUR", "1h", 10), ExchangeError, [], "ExchangeError in fetch_ohlcv"),
            ("create_market_buy_order", ("ETH/EUR", 0.01), InsufficientFunds, None, "InsufficientFunds in create_market_buy_order"),
            ("create_market_buy_order", ("ETH/EUR", 0.01), InvalidOrder, None, "InvalidOrder in create_market_buy_order"),
            ("create_market_sell_order", ("BTC/EUR", 0.005), BadSymbol, None, "BadSymbol in create_market_sell_order"),
            ("create_market_sell_order", ("BTC/EUR", 0.005), RequestTimeout, None, "RequestTimeout in create_market_sell_order"),
        ]
    )
    async def test_method_error_handling(
        self, mock_bitvavo_exchange_fixture, caplog,
        method_name, method_args, ccxt_error, expected_return, log_message_part
    ):
        caplog.set_level(logging.WARNING) # Errors are logged as warning or error
        mock_exchange = mock_bitvavo_exchange_fixture

        # Configure the mocked exchange method to raise the specified CCXT error
        mocked_method = AsyncMock(side_effect=ccxt_error(str(ccxt_error))) # Pass error message for context
        setattr(mock_exchange, method_name, mocked_method)

        executor = BitvavoExecutor()

        # Call the BitvavoExecutor method
        result = await getattr(executor, method_name)(*method_args)

        assert result == expected_return
        assert log_message_part in caplog.text
        getattr(mock_exchange, method_name).assert_called_once()




# Imports for BiasReflector tests
from core.bias_reflector import BiasReflector, BIAS_COOLDOWN_SECONDS
# datetime, timedelta, os, json, asyncio, patch are already imported.
# pytest is available.

@pytest.fixture
def bias_reflector_instance_with_temp_memory(tmp_path):
    """
    Provides a BiasReflector instance using a temporary BIAS_MEMORY_FILE.
    Ensures test isolation by providing a clean memory file for each test.
    """
    temp_memory_dir = tmp_path / "bias_memory_test"
    temp_memory_dir.mkdir()
    temp_bias_file_path = temp_memory_dir / "bias_memory.json"

    # Patch the BIAS_MEMORY_FILE constant in the bias_reflector module
    with patch('core.bias_reflector.BIAS_MEMORY_FILE', str(temp_bias_file_path)):
        reflector = BiasReflector()
        yield reflector, str(temp_bias_file_path) # Yield both instance and path for optional direct checks

    # Cleanup is handled by tmp_path fixture for the directory and its contents.

class TestBiasReflector:
    TEST_TOKEN = "TEST/TOKEN"
    TEST_STRATEGY = "TestStrategy"

    @pytest.mark.asyncio
    async def test_initial_bias_score(self, bias_reflector_instance_with_temp_memory):
        """Checks the default bias for a new token/strategy."""
        bias_reflector, _ = bias_reflector_instance_with_temp_memory
        initial_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        assert initial_bias == 0.5, "Initial bias for a new token/strategy should be 0.5 (neutral)."

    @pytest.mark.asyncio
    async def test_update_bias_winning_trade(self, bias_reflector_instance_with_temp_memory, caplog):
        """Simulates a winning trade and checks if bias updates positively."""
        caplog.set_level(logging.INFO)
        bias_reflector, memory_file_path = bias_reflector_instance_with_temp_memory

        initial_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY) # Should be 0.5

        ai_suggested_bias = 0.7
        ai_confidence = 0.8
        trade_profit_pct = 0.02 # 2% profit

        await bias_reflector.update_bias(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            new_ai_bias=ai_suggested_bias,
            confidence=ai_confidence,
            trade_result_pct=trade_profit_pct
        )

        updated_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert updated_bias > initial_bias, "Bias should increase after a winning trade with positive AI signals."
        # The exact calculation:
        # current_bias = 0.5
        # trade_influence_factor = 0.8 * 0.1 = 0.08
        # target_bias = 0.7 (since new_ai_bias > current_bias)
        # adjustment = (0.7 - 0.5) * 0.08 * (1 + min(0.02 * 10, 1))
        #            = 0.2 * 0.08 * (1 + 0.2) = 0.2 * 0.08 * 1.2 = 0.016 * 1.2 = 0.0192
        # calculated_new_bias = 0.5 + 0.0192 = 0.5192
        assert abs(updated_bias - 0.5192) < 0.0001, f"Bias calculation mismatch. Expected ~0.5192, got {updated_bias}"
        assert f"Bias voor {self.TEST_TOKEN}/{self.TEST_STRATEGY} bijgewerkt naar {updated_bias:.3f}" in caplog.text

        # Verify memory file content
        with open(memory_file_path, 'r') as f:
            memory_data = json.load(f)
        assert memory_data[self.TEST_TOKEN][self.TEST_STRATEGY]['bias'] == updated_bias
        assert memory_data[self.TEST_TOKEN][self.TEST_STRATEGY]['trade_count'] == 1
        assert memory_data[self.TEST_TOKEN][self.TEST_STRATEGY]['total_profit_pct'] == trade_profit_pct

    @pytest.mark.asyncio
    async def test_update_bias_losing_trade(self, bias_reflector_instance_with_temp_memory, caplog):
        """Simulates a losing trade and checks if bias updates appropriately."""
        caplog.set_level(logging.INFO)
        bias_reflector, _ = bias_reflector_instance_with_temp_memory
        initial_bias = 0.6 # Set a slightly positive initial bias for better testing of decrease
        bias_reflector.bias_memory[self.TEST_TOKEN] = {self.TEST_STRATEGY: {'bias': initial_bias, 'last_update': None, 'trade_count': 0, 'total_profit_pct': 0.0}}


        ai_suggested_bias = 0.3
        ai_confidence = 0.6
        trade_profit_pct = -0.01 # -1% loss

        await bias_reflector.update_bias(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            new_ai_bias=ai_suggested_bias,
            confidence=ai_confidence,
            trade_result_pct=trade_profit_pct
        )
        updated_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert updated_bias < initial_bias, "Bias should decrease after a losing trade, especially with negative AI signals."
        # current_bias = 0.6
        # trade_influence_factor = 0.6 * 0.1 = 0.06
        # target_bias = 0.3 (since new_ai_bias < current_bias)
        # adjustment = (0.3 - 0.6) * 0.06 * (1 + min(abs(-0.01) * 10, 1))
        #            = -0.3 * 0.06 * (1 + 0.1) = -0.3 * 0.06 * 1.1 = -0.018 * 1.1 = -0.0198
        # calculated_new_bias = 0.6 - 0.0198 = 0.5802
        assert abs(updated_bias - 0.5802) < 0.0001, f"Bias calculation mismatch. Expected ~0.5802, got {updated_bias}"
        assert f"Bias voor {self.TEST_TOKEN}/{self.TEST_STRATEGY} bijgewerkt naar {updated_bias:.3f}" in caplog.text

    @pytest.mark.asyncio
    async def test_update_bias_ai_only(self, bias_reflector_instance_with_temp_memory, caplog):
        """Simulates an update based only on AI reflection (no trade)."""
        caplog.set_level(logging.INFO)
        bias_reflector, _ = bias_reflector_instance_with_temp_memory
        initial_bias = 0.5

        ai_suggested_bias = 0.9
        ai_confidence = 0.9

        await bias_reflector.update_bias(
            self.TEST_TOKEN, self.TEST_STRATEGY,
            new_ai_bias=ai_suggested_bias,
            confidence=ai_confidence
            # trade_result_pct is None
        )
        updated_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Expected: (current_bias * (1 - confidence) + new_ai_bias * confidence)
        #         = (0.5 * (1 - 0.9) + 0.9 * 0.9)
        #         = (0.5 * 0.1 + 0.81) = 0.05 + 0.81 = 0.86
        expected_calculated_bias = 0.86
        assert abs(updated_bias - expected_calculated_bias) < 0.0001, "Bias should move towards AI suggestion, weighted by confidence."
        assert f"Bias voor {self.TEST_TOKEN}/{self.TEST_STRATEGY} bijgewerkt naar {updated_bias:.3f}" in caplog.text

    @pytest.mark.asyncio
    async def test_bias_update_cooldown(self, bias_reflector_instance_with_temp_memory, caplog):
        """Checks that bias is not updated during the cooldown period."""
        caplog.set_level(logging.DEBUG) # Need DEBUG for cooldown message
        bias_reflector, _ = bias_reflector_instance_with_temp_memory

        # First update to set 'last_update' timestamp
        await bias_reflector.update_bias(self.TEST_TOKEN, self.TEST_STRATEGY, 0.6, 0.7)
        first_updated_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Attempt another update immediately (should be blocked by cooldown)
        await bias_reflector.update_bias(self.TEST_TOKEN, self.TEST_STRATEGY, 0.1, 0.9, trade_result_pct=-0.05)
        bias_after_cooldown_attempt = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        assert abs(bias_after_cooldown_attempt - first_updated_bias) < 0.0001, "Bias should not change during cooldown period."
        assert f"Cooldown actief voor {self.TEST_TOKEN}/{self.TEST_STRATEGY}" in caplog.text

    @pytest.mark.asyncio
    async def test_bias_update_after_cooldown_mocked_time(self, bias_reflector_instance_with_temp_memory, caplog):
        """Tests that bias updates after cooldown period by mocking datetime.now()."""
        caplog.set_level(logging.INFO)
        bias_reflector, _ = bias_reflector_instance_with_temp_memory

        # Initial update
        initial_update_time = datetime(2023, 1, 1, 12, 0, 0)
        with patch('core.bias_reflector.datetime') as mock_datetime:
            mock_datetime.now.return_value = initial_update_time
            mock_datetime.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts) # Ensure fromisoformat still works

            await bias_reflector.update_bias(self.TEST_TOKEN, self.TEST_STRATEGY, 0.6, 0.7)

        first_updated_bias = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)

        # Attempt update during cooldown (mock time slightly after initial update)
        with patch('core.bias_reflector.datetime') as mock_datetime_cooldown:
            mock_datetime_cooldown.now.return_value = initial_update_time + timedelta(seconds=BIAS_COOLDOWN_SECONDS / 2)
            mock_datetime_cooldown.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)

            await bias_reflector.update_bias(self.TEST_TOKEN, self.TEST_STRATEGY, 0.2, 0.8)
            bias_during_cooldown = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)
            assert abs(bias_during_cooldown - first_updated_bias) < 0.0001, "Bias should not update during mocked cooldown."

        # Attempt update after cooldown (mock time well after cooldown period)
        with patch('core.bias_reflector.datetime') as mock_datetime_after_cooldown:
            mock_datetime_after_cooldown.now.return_value = initial_update_time + timedelta(seconds=BIAS_COOLDOWN_SECONDS + 1)
            mock_datetime_after_cooldown.fromisoformat.side_effect = lambda ts: datetime.fromisoformat(ts)

            await bias_reflector.update_bias(self.TEST_TOKEN, self.TEST_STRATEGY, 0.2, 0.8) # Different values to ensure change

        bias_after_cooldown = bias_reflector.get_bias_score(self.TEST_TOKEN, self.TEST_STRATEGY)
        assert bias_after_cooldown != first_updated_bias, "Bias should update after cooldown period."
        # Expected: (0.42 * (1-0.8) + 0.2 * 0.8) = (0.42 * 0.2 + 0.16) = 0.084 + 0.16 = 0.244
        # Current bias before this update was (0.5 * (1-0.7) + 0.6*0.7) = 0.15 + 0.42 = 0.57
        # So, (0.57 * 0.2 + 0.2 * 0.8) = 0.114 + 0.16 = 0.274
        assert abs(bias_after_cooldown - 0.274) < 0.0001, f"Bias calculation error after cooldown. Expected ~0.274, got {bias_after_cooldown}"
        assert f"Bias voor {self.TEST_TOKEN}/{self.TEST_STRATEGY} bijgewerkt naar {bias_after_cooldown:.3f}" in caplog.text



# Imports for Backtester tests
from core.backtester import Backtester #, TIMEFRAME_TO_SECONDS
from core.bitvavo_executor import BitvavoExecutor # For mocking
# ParamsManager, CNNPatterns are already imported
from datetime import timezone # Already imported datetime, just need timezone

@pytest.fixture
def mock_backtester_dependencies():
    """Provides mocked dependencies for Backtester."""
    mock_params_manager = MagicMock(spec=ParamsManager)
    mock_cnn_detector = MagicMock(spec=CNNPatterns)
    mock_bitvavo_executor = MagicMock(spec=BitvavoExecutor)

    # Setup default return values for ParamsManager global params
    mock_params_manager.get_global_params.return_value = {
        'perform_backtesting': True,
        'backtest_start_date_str': "2023-01-01",
        'backtest_entry_threshold': 0.7,
        'backtest_take_profit_pct': 0.05,
        'backtest_stop_loss_pct': 0.02,
        'backtest_hold_duration_candles': 20,
        'backtest_initial_capital': 1000.0,
        'backtest_stake_pct_capital': 0.1
    }
    return mock_params_manager, mock_cnn_detector, mock_bitvavo_executor

def create_mock_ohlcv_data(start_date_str: str, num_candles: int, timeframe: str) -> pd.DataFrame:
    """Helper to create a Pandas DataFrame with mock OHLCV data."""
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Use TIMEFRAME_TO_SECONDS from core.backtester or define a local version for tests
    TIMEFRAME_TO_SECONDS_TEST = {
        '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400
    }
    interval_seconds = TIMEFRAME_TO_SECONDS_TEST.get(timeframe, 3600) # Default to 1h if not found

    dates = [start_dt + timedelta(seconds=i * interval_seconds) for i in range(num_candles)]
    data = {
        'open': np.random.uniform(90, 110, num_candles),
        'high': np.random.uniform(100, 120, num_candles),
        'low': np.random.uniform(80, 100, num_candles),
        'close': np.random.uniform(90, 110, num_candles),
        'volume': np.random.uniform(1000, 5000, num_candles)
    }
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='date'))
    # Ensure high is max and low is min of open/close for realistic candles
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 5, num_candles)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 5, num_candles)
    df.attrs['pair'] = 'TEST/USDT'
    df.attrs['timeframe'] = timeframe
    return df

class TestBacktester:
    @pytest.mark.asyncio
    async def test_run_backtest_scenario(self, mock_backtester_dependencies, caplog):
        caplog.set_level(logging.INFO)
        mock_params_manager, mock_cnn_detector, mock_bitvavo_executor = mock_backtester_dependencies

        # 1. Configure mock BitvavoExecutor to return predefined OHLCV data
        mock_ohlcv_df = create_mock_ohlcv_data("2023-01-01", 200, "1h") # 200 candles for 1h timeframe
        # _fetch_backtest_data in Backtester uses fetch_ohlcv in a loop.
        # For simplicity, we'll mock _fetch_backtest_data directly.
        # If we wanted to test the looping logic in _fetch_backtest_data, we'd mock fetch_ohlcv on BitvavoExecutor.

        # 2. Configure mock CNNPatterns
        # Simulate a sequence of pattern scores. Let's make it simple: always above threshold for a few candles.
        # predict_pattern_score is called with a DataFrame window.
        # We need to ensure the mock can be called repeatedly.
        # Let's make it return a high score for the first few calls, then lower.
        pattern_scores_sequence = [0.8, 0.85, 0.75, 0.6, 0.5] # Example sequence
        mock_cnn_detector.predict_pattern_score = MagicMock(side_effect=pattern_scores_sequence + [0.5] * 100) # Default to low score after sequence

        # Instantiate Backtester with mocked dependencies
        backtester = Backtester(
            params_manager=mock_params_manager,
            cnn_pattern_detector=mock_cnn_detector,
            bitvavo_executor=mock_bitvavo_executor
        )

        # Patch _fetch_backtest_data to return our mock OHLCV data directly
        # This avoids dealing with the complexities of mocking the paginated fetch_ohlcv calls.
        with patch.object(backtester, '_fetch_backtest_data', AsyncMock(return_value=mock_ohlcv_df.copy())) as mock_fetch_data, \
             patch.object(backtester, '_save_backtest_results', MagicMock()) as mock_save_results:

            # Define backtest parameters
            test_symbol = "TEST/USDT"
            test_timeframe = "1h"
            test_pattern_type = "test_pattern"
            test_architecture_key = "test_arch"
            test_sequence_length = 30 # Must be less than num_candles in mock_ohlcv_df - indicator period (e.g. 20 for BBANDS)

            # Run the backtest
            results = await backtester.run_backtest(
                symbol=test_symbol,
                timeframe=test_timeframe,
                pattern_type=test_pattern_type,
                architecture_key=test_architecture_key,
                sequence_length=test_sequence_length
            )

            # Assertions
            assert results is not None, "run_backtest should return results."
            mock_fetch_data.assert_called_once_with(test_symbol, test_timeframe, "2023-01-01")

            assert 'metrics' in results, "Results should contain 'metrics'."
            assert 'trades' in results, "Results should contain 'trades'."
            assert 'portfolio_history' in results, "Results should contain 'portfolio_history'."

            # Example basic assertions on metrics (more detailed assertions require specific data crafting)
            assert isinstance(results['metrics']['final_capital'], float)
            assert isinstance(results['metrics']['total_return_pct'], float)
            assert isinstance(results['metrics']['num_trades'], int)
            assert isinstance(results['metrics']['win_rate_pct'], float)

            # Based on pattern_scores_sequence [0.8, 0.85, 0.75, 0.6, 0.5] and entry_threshold 0.7
            # We expect 3 entries if positions are closed before next signal.
            # The exact number of trades depends on TP, SL, hold duration.
            # For this initial test, let's check if at least one trade happened due to high scores.
            if any(s >= mock_params_manager.get_global_params()['backtest_entry_threshold'] for s in pattern_scores_sequence):
                 assert results['metrics']['num_trades'] > 0, "Expected at least one trade based on mock scores."
            else:
                 assert results['metrics']['num_trades'] == 0, "Expected no trades if all scores are below threshold."


            mock_save_results.assert_called_once()
            # We can also check the content passed to _save_backtest_results if needed.
            # saved_data = mock_save_results.call_args[0][0]
            # assert saved_data['symbol'] == test_symbol

            # Check if _add_indicators was called (implicitly via run_backtest -> _fetch_backtest_data -> _add_indicators)
            # This is harder to check directly without more patching of _add_indicators itself or checking logs.
            # For now, we assume if data is processed and trades occur, indicators were likely involved.
            assert "Backtest finished" in caplog.text
            assert f"Initial Capital: {mock_params_manager.get_global_params()['backtest_initial_capital']:.2f}" in caplog.text



# Imports for ExitOptimizer tests
from core.exit_optimizer import ExitOptimizer
# Most dependencies are already imported or can be patched via core.exit_optimizer.*

# Imports for AIActivationEngine tests
from core.ai_activation_engine import AIActivationEngine
# ReflectieLus and CNNPatterns might be needed if not already imported or if specific mocks are used.
# from core.reflectie_lus import ReflectieLus # Already imported by other tests or mocked
# from core.cnn_patterns import CNNPatterns # Already imported by other tests or mocked
import pandas as pd # Already imported
import numpy as np # Already imported
from datetime import datetime, timedelta # Already imported
# import os # Already imported
# import json # Already imported
import logging # Already imported
# import asyncio # Already imported
# import sys # Already imported
# from unittest.mock import patch, AsyncMock # Already imported

# Helper function to create mock data for AIActivationEngine tests
def create_mock_dataframe_for_ai_activation(timeframe: str = '5m', num_candles: int = 100) -> pd.DataFrame:
    data = []
    now = datetime.utcnow()
    interval_seconds_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
    interval_seconds = interval_seconds_map.get(timeframe, 300)
    for i in range(num_candles):
        date = now - timedelta(seconds=(num_candles - 1 - i) * interval_seconds)
        open_ = 100 + i * 0.01 + np.random.randn() * 0.1
        close_ = open_ + np.random.randn() * 0.5
        high_ = max(open_, close_) + np.random.rand() * 0.2
        low_ = min(open_, close_) - np.random.rand() * 0.2
        volume = 1000 + np.random.rand() * 2000
        data.append([date, open_, high_, low_, close_, volume])
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.attrs['timeframe'] = timeframe
    df.attrs['pair'] = 'ETH/USDT' # Example pair
    return df

# Mock classes for AIActivationEngine tests
class MockReflectieLusForTest:
    def __init__(self):
        self.last_prompt_type = None
        self.last_pattern_data = None
        self.last_symbol = None
        self.last_trade_context = None
        self.call_count = 0

    async def process_reflection_cycle(self, **kwargs):
        self.call_count += 1
        self.last_prompt_type = kwargs.get('prompt_type')
        self.last_pattern_data = kwargs.get('pattern_data')
        self.last_symbol = kwargs.get('symbol')
        self.last_trade_context = kwargs.get('trade_context')
        logging.info(f"MockReflectieLusForTest.process_reflection_cycle called for {self.last_symbol} with prompt_type {self.last_prompt_type}")
        return {
            "timestamp": datetime.now().isoformat(),
            "token": self.last_symbol,
            "strategyId": kwargs.get('strategy_id'),
            "prompt_type": self.last_prompt_type,
            "summary": "Mocked reflection cycle completed.",
            "mode": kwargs.get('mode'),
            "pattern_data_used": self.last_pattern_data is not None,
            "received_trade_context": self.last_trade_context
        }

class MockBiasReflector:
    def get_bias_score(self, token, strategy_id): return 0.65

class MockConfidenceEngine:
    def get_confidence_score(self, token, strategy_id): return 0.75

class MockCNNPatternsForAIAE(CNNPatterns): # Extend the actual class or use a more specific mock
    def __init__(self): # Add init if CNNPatterns has one that needs to be called or mocked
        super().__init__() # Call parent __init__ if necessary
        self.mock_pattern_keys = ["bullFlag", "bearFlag", "breakout", "rsiDivergence", "CDLDOJI", "doubleTop", "headAndShoulders"]

    def get_all_detectable_pattern_keys(self) -> list:
        return self.mock_pattern_keys

    async def detect_patterns_multi_timeframe(self, candles_by_timeframe: dict, symbol: str) -> dict:
        # Return a structure that activate_ai expects, including 'cnn_predictions' and 'context'
        return {
            "cnn_predictions": {
                "5m_bullFlag_score": 0.8, # Example, make it configurable if needed for tests
                "1h_bearFlag_score": 0.75
            },
            "patterns": {"detected_rule_pattern": True}, # Example
            "context": {"volume_spike": True, "trend_strength": 0.6} # Example
        }

class MockPromptBuilderForAIAE: # Renamed to avoid conflict
    async def generate_prompt_with_data(self, **kwargs):
        logging.info(f"MockPromptBuilderForAIAE.generate_prompt_with_data called for {kwargs.get('symbol')}")
        return f"Fallback mock prompt for {kwargs.get('symbol')} - {kwargs.get('prompt_type')}"


@pytest.fixture
def mock_reflectie_lus_instance():
    return MockReflectieLusForTest()

@pytest.fixture
def ai_activation_engine_instance(mock_reflectie_lus_instance):
    # engine = AIActivationEngine(reflectie_lus_instance=mock_reflectie_lus_instance)
    # # We need to mock dependencies of AIActivationEngine's __init__ if they are complex
    # # For now, assume PromptBuilder, GrokSentimentFetcher can be default or None for these tests,
    # # or patch them globally if they cause issues.
    with patch('core.ai_activation_engine.PromptBuilder', new_callable=MockPromptBuilderForAIAE) as mock_pb, \
         patch('core.ai_activation_engine.GrokSentimentFetcher') as mock_gsf, \
         patch.object(AIActivationEngine, 'cnn_patterns_detector', new_callable=MockCNNPatternsForAIAE) as mock_cnn_detector_attr:
        # mock_cnn_detector_attr = MockCNNPatternsForAIAE() # This line should be removed as patch.object handles assignment
        engine = AIActivationEngine(reflectie_lus_instance=mock_reflectie_lus_instance)
        # If AIActivationEngine initializes its own CNNPatterns, we need to mock that specific instance or the class.
        # The patch.object for 'cnn_patterns_detector' on the class should handle this.
        # If GrokSentimentFetcher is initialized and fails without API keys, mock it to return a dummy.
        mock_gsf.return_value.fetch_live_search_data = AsyncMock(return_value=[]) # Default no sentiment
        return engine


# --- Tests for AIActivationEngine ---

@pytest.mark.asyncio
async def test_activate_ai_entry_signal(ai_activation_engine_instance, mock_reflectie_lus_instance):
    engine = ai_activation_engine_instance
    test_token = "ETH/USDT"
    test_strategy_id = "DUOAI_Strategy_Entry"
    mock_candles_by_timeframe = {
        '5m': create_mock_dataframe_for_ai_activation('5m', 60),
        '1h': create_mock_dataframe_for_ai_activation('1h', 60)
    }
    entry_trade_context = {"signal_strength": 0.85, "some_other_entry_info": "test_value"}

    # Mock _should_trigger_ai to ensure AI activates for this test
    with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=True):
        entry_reflection = await engine.activate_ai(
            trigger_type='entry_signal',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=entry_trade_context,
            mode='dry_run', # Example mode
            bias_reflector_instance=MockBiasReflector(),
            confidence_engine_instance=MockConfidenceEngine()
        )

    assert entry_reflection is not None, "Entry signal reflection was not triggered or returned None."
    assert mock_reflectie_lus_instance.last_prompt_type == 'entry_analysis', \
        f"Incorrect prompt_type for entry_signal: {mock_reflectie_lus_instance.last_prompt_type}"
    assert mock_reflectie_lus_instance.last_pattern_data is not None, "Pattern data not passed for entry_signal"
    assert mock_reflectie_lus_instance.last_trade_context.get("signal_strength") == 0.85, "Trade context not passed correctly"

@pytest.mark.asyncio
async def test_activate_ai_trade_closed(ai_activation_engine_instance, mock_reflectie_lus_instance):
    engine = ai_activation_engine_instance
    test_token = "BTC/USDT"
    test_strategy_id = "DUOAI_Strategy_Exit"
    mock_candles_by_timeframe = {'15m': create_mock_dataframe_for_ai_activation('15m')}
    closed_trade_context = {"entry_price": 2500, "exit_price": 2450, "profit_pct": -0.02, "trade_id": "t123"}

    with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=True):
        exit_reflection = await engine.activate_ai(
            trigger_type='trade_closed',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=closed_trade_context,
            mode='live',
            bias_reflector_instance=MockBiasReflector(),
            confidence_engine_instance=MockConfidenceEngine()
        )

    assert exit_reflection is not None, "Trade closed reflection was not triggered."
    assert mock_reflectie_lus_instance.last_prompt_type == 'post_trade_analysis', \
        f"Incorrect prompt_type for trade_closed: {mock_reflectie_lus_instance.last_prompt_type}"
    assert mock_reflectie_lus_instance.last_trade_context.get("trade_id") == "t123", "Trade context not passed correctly"

@pytest.mark.asyncio
async def test_activate_ai_cnn_pattern_detected(ai_activation_engine_instance, mock_reflectie_lus_instance):
    engine = ai_activation_engine_instance
    test_token = "ADA/USDT"
    test_strategy_id = "CNN_Strategy"
    mock_candles_by_timeframe = {'1h': create_mock_dataframe_for_ai_activation('1h')}
    cnn_context = {"pattern_name": "bull_flag_1h", "confidence": 0.92}

    with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=True):
        cnn_reflection = await engine.activate_ai(
            trigger_type='cnn_pattern_detected',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=cnn_context,
            mode='live',
            bias_reflector_instance=MockBiasReflector(),
            confidence_engine_instance=MockConfidenceEngine()
        )
    assert cnn_reflection is not None, "CNN pattern reflection was not triggered."
    assert mock_reflectie_lus_instance.last_prompt_type == 'pattern_analysis', \
        f"Incorrect prompt_type for cnn_pattern_detected: {mock_reflectie_lus_instance.last_prompt_type}"
    assert mock_reflectie_lus_instance.last_trade_context.get("pattern_name") == "bull_flag_1h", "Trade context not passed correctly"


@pytest.mark.asyncio
async def test_activate_ai_unknown_trigger(ai_activation_engine_instance, mock_reflectie_lus_instance):
    engine = ai_activation_engine_instance
    test_token = "SOL/USDT"
    test_strategy_id = "GeneralPurposeStrategy"
    mock_candles_by_timeframe = {'5m': create_mock_dataframe_for_ai_activation('5m')}
    unknown_context = {"detail": "some_random_event_for_general_analysis"}

    with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=True):
        unknown_reflection = await engine.activate_ai(
            trigger_type='unknown_signal',
            token=test_token,
            candles_by_timeframe=mock_candles_by_timeframe,
            strategy_id=test_strategy_id,
            trade_context=unknown_context,
            mode='live',
            bias_reflector_instance=MockBiasReflector(),
            confidence_engine_instance=MockConfidenceEngine()
        )
    assert unknown_reflection is not None, "Unknown signal reflection was not triggered."
    assert mock_reflectie_lus_instance.last_prompt_type == 'general_analysis', \
        f"Incorrect prompt_type for unknown_signal: {mock_reflectie_lus_instance.last_prompt_type}"
    assert mock_reflectie_lus_instance.last_trade_context.get("detail") == "some_random_event_for_general_analysis"

@pytest.mark.asyncio
async def test_pattern_score_val_calculation_in_activate_ai(ai_activation_engine_instance):
    engine = ai_activation_engine_instance # Uses the fixture with MockCNNPatternsForAIAE

    # Configure the mock CNN detector on the specific engine instance for this test if needed,
    # or ensure the fixture's mock is adequate.
    # The fixture uses MockCNNPatternsForAIAE which defines get_all_detectable_pattern_keys
    # and detect_patterns_multi_timeframe.

    # Example: Override specific mock behavior if necessary
    class CustomMockCNNPatterns(MockCNNPatternsForAIAE):
        def get_all_detectable_pattern_keys(self) -> list:
            return ["bullFlag", "bearTrap", "futurePattern"] # 3 total possible

        async def detect_patterns_multi_timeframe(self, candles_by_timeframe: dict, symbol: str) -> dict:
            return {
                "cnn_predictions": {
                    "5m_bullFlag_score": 0.85,  # Detected (>= 0.7 threshold)
                    "1h_bullFlag_score": 0.6,   # Not detected
                    "5m_bearTrap_score": 0.5,   # Not detected
                    "15m_nonCnnPattern_score": 0.9, # Not in get_all_detectable_pattern_keys
                    "1h_futurePattern_score": 0.7 # Detected (>= 0.7 threshold)
                }, "patterns": {}, "context": {}
            }
    engine.cnn_patterns_detector = CustomMockCNNPatterns() # Override the fixture's detector for this test
    engine.CNN_DETECTION_THRESHOLD = 0.7 # Ensure threshold is known for the test

    calculated_pattern_score = None
    # Patch _should_trigger_ai to prevent full AI activation, and to capture trigger_data
    with patch.object(engine, '_should_trigger_ai', new_callable=AsyncMock, return_value=False) as mock_should_trigger:
        await engine.activate_ai(
            trigger_type='test_pattern_score_trigger',
            token='TEST/PATTERNSCORE',
            candles_by_timeframe={}, # Mocked, not used by CustomMockCNNPatterns
            strategy_id='test_pattern_score_strat',
            bias_reflector_instance=MockBiasReflector(),
            confidence_engine_instance=MockConfidenceEngine(),
            mode='test'
        )
        assert mock_should_trigger.called, "_should_trigger_ai was not called"
        trigger_data_arg = mock_should_trigger.call_args[0][0]
        calculated_pattern_score = trigger_data_arg.get('patternScore')

    assert calculated_pattern_score is not None, "patternScore was not found in trigger_data."
    # Expected: bullFlag (1), futurePattern (1) = 2 detected. Total possible = 3. Score = 2/3.
    expected_score = 2/3
    assert abs(calculated_pattern_score - expected_score) < 0.001, \
        f"Calculated pattern_score_val {calculated_pattern_score} does not match expected {expected_score}"

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_id, sentiment_return, trigger_data_input, expected_trigger_result, expected_log_part",
    [
        ("SentimentBoostsTrigger", [{"item": "data"}], # Grok returns sentiment
         {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0},
         True, "Sentiment items found"), # Expected: score 1 (sentiment) + 2 (low conf) = 3. Threshold for conf 0.3 is 2. 3 >= 2 -> True

        ("NoSentimentLowConfStillTriggers", [], # Grok returns no sentiment
         {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0},
         True, "No sentiment items found"), # Expected: score 0 (no sentiment) + 2 (low conf) = 2. Threshold for conf 0.3 is 2. 2 >= 2 -> True

        ("NoSentimentHighConfNoTrigger", [],
         {"patternScore": 0.1, "volumeSpike": False, "learned_confidence": 0.8, "time_since_last_reflection": 0, "profit_metric": 0.0},
         False, "No sentiment items found"), # Expected: score 0. Threshold for conf 0.8 is 3. 0 < 3 -> False

        ("GrokFetcherNoneLowConfTriggers", None, # Grok fetcher itself is None
         {"patternScore": 0.5, "volumeSpike": False, "learned_confidence": 0.3, "time_since_last_reflection": 0, "profit_metric": 0.0},
         True, "GrokSentimentFetcher not available"), # Expected: score 0 (no sentiment) + 2 (low conf) = 2. Threshold for conf 0.3 is 2. 2 >= 2 -> True
    ]
)
async def test_should_trigger_ai_logic_detailed(
    ai_activation_engine_instance, caplog,
    test_id, sentiment_return, trigger_data_input, expected_trigger_result, expected_log_part
):
    engine = ai_activation_engine_instance
    caplog.set_level(logging.DEBUG) # Capture DEBUG logs from _should_trigger_ai

    original_fetcher = engine.grok_sentiment_fetcher
    if sentiment_return is None and original_fetcher is not None: # Scenario: GrokFetcher is None
        engine.grok_sentiment_fetcher = None
    elif original_fetcher is not None : # sentiment_return is list (empty or with data)
        engine.grok_sentiment_fetcher.fetch_live_search_data = AsyncMock(return_value=sentiment_return)


    result = await engine._should_trigger_ai(trigger_data_input, "TEST/TOKEN", "live")
    assert result == expected_trigger_result, f"Test ID '{test_id}': Trigger result mismatch."
    assert expected_log_part in caplog.text, f"Test ID '{test_id}': Expected log string '{expected_log_part}' not found."

    engine.grok_sentiment_fetcher = original_fetcher # Restore fetcher


def get_exit_optimizer_param_side_effect(scenario_params):
    """
    Returns a side_effect function for mock_params_manager.get_param
    for ExitOptimizer tests.
    """
    def side_effect(key, strategy_id=None, default=None):
        # ParamsManager.get_param(key, strategy_id=current_strategy_id, default=...)
        val = scenario_params.get(key, default)
        # print(f"ExitParamsManager.get_param(key='{key}', strategy_id='{strategy_id}') -> returning: {val}, default was: {default}")
        return val
    return side_effect

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_name, mock_params, mock_cnn_output_key, mock_cnn_score, mock_rule_patterns, ai_intent_gpt, ai_intent_grok, ai_conf_gpt, ai_conf_grok, trade_profit_pct, expected_exit, expected_reason_part",
    [
        # Scenario 1: Exit Approved - Strong Bearish Pattern Score Triggers Exit (CNN part)
        (
            "Exit_StrongBearish_CNN_AIConfirm", # Scenario name
            { # mock_params for ParamsManager
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.0, # No rule-based contribution for this sub-scenario
                "strongPatternThreshold": 0.7,
                "exitPatternConfThreshold": 0.5,
                "exitConvictionDropTrigger": 0.3,
                "minProfitForLowConfExit": 0.005,
                "exitAISellIntentConfThreshold": 0.6,
            },
            "5m_no_bullFlag_score", 0.9, # mock_cnn_output_key, mock_cnn_score. Weighted: 0.9 * 0.8 = 0.72. >= 0.7 (strong)
            {}, # mock_rule_patterns
            "HOLD", "HOLD", # ai_intent_gpt, ai_intent_grok -> ai_exit_intent = False
            0.55, 0.55,     # ai_conf_gpt, ai_conf_grok -> combined_confidence = 0.55. (0.55 > exitPatternConfThreshold (0.5))
            0.10,           # trade_profit_pct (profitable)
            True,           # expected_exit
            "bearish_pattern_with_ai_confirmation", # expected_reason_part
        ),
        # Scenario 1b: Exit Approved - Strong Bearish Pattern Score Triggers Exit (Rule-based part)
        (
            "Exit_StrongBearish_Rule_AIConfirm",
            {
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.9, # Rule score 0.9. Weighted: 0.9 * 0.8 = 0.72. >= 0.7 (strong)
                "strongPatternThreshold": 0.7,
                "exitPatternConfThreshold": 0.5,
                "exitConvictionDropTrigger": 0.3, "minProfitForLowConfExit": 0.005, "exitAISellIntentConfThreshold": 0.6,
            },
            "5m_no_bullFlag_score", 0.0, # No CNN contribution
            {"bearishEngulfing": True}, # Strong rule-based pattern
            "HOLD", "HOLD", 0.55, 0.55,
            0.10, True, "bearish_pattern_with_ai_confirmation",
        ),
        # Scenario 2: No Exit - Weighted Bearish Score Below strongPatternThreshold
        (
            "NoExit_WeakBearishPattern",
            {
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.0,
                "strongPatternThreshold": 0.7, # Weighted score (0.32) < 0.7
                "exitPatternConfThreshold": 0.5,
                "exitConvictionDropTrigger": 0.3, "minProfitForLowConfExit": 0.005, "exitAISellIntentConfThreshold": 0.6,
            },
            "5m_no_bullFlag_score", 0.4, # mock_cnn_score. Weighted: 0.4 * 0.8 = 0.32
            {}, # No rule patterns
            "HOLD", "HOLD",
            0.8, 0.8,       # High AI confidence, but doesn't matter as pattern is weak and intent is HOLD
            0.10,
            False,          # expected_exit
            "no_ai_exit_signal",
        ),
    ]
)
@patch('core.exit_optimizer.GPTReflector')
@patch('core.exit_optimizer.GrokReflector')
@patch('core.exit_optimizer.PromptBuilder')
@patch('core.exit_optimizer.ParamsManager')
@patch('core.exit_optimizer.CNNPatterns')
# Mock BiasReflector and ConfidenceEngine at the class level if their __init__ is problematic without further setup,
# even if not directly used in should_exit paths being tested.
# For now, assuming their __init__ is simple or their instances are not problematic if unused.
async def test_exit_optimizer_should_exit(
    MockCNNPatterns, MockParamsManager, MockPromptBuilder, MockGrokReflector, MockGPTReflector,
    scenario_name, mock_params, mock_cnn_output_key, mock_cnn_score, mock_rule_patterns,
    ai_intent_gpt, ai_intent_grok, ai_conf_gpt, ai_conf_grok,
    trade_profit_pct, expected_exit, expected_reason_part
):
    # Setup mocks for instances
    mock_params_manager_instance = MockParamsManager.return_value
    mock_params_manager_instance.get_param.side_effect = get_exit_optimizer_param_side_effect(mock_params)

    mock_cnn_patterns_instance = MockCNNPatterns.return_value
    # Construct the cnn_output based on parameterized inputs
    cnn_output = {"cnn_predictions": {}, "patterns": mock_rule_patterns}
    if mock_cnn_score > 0: # Only add if score is meant to be present
        cnn_output["cnn_predictions"][mock_cnn_output_key] = mock_cnn_score
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.return_value = cnn_output

    mock_gpt_reflector_instance = MockGPTReflector.return_value
    mock_gpt_reflector_instance.ask_ai.return_value = {
        "intentie": ai_intent_gpt, "confidence": ai_conf_gpt, "reflectie": "GPT Test"
    }

    mock_grok_reflector_instance = MockGrokReflector.return_value
    mock_grok_reflector_instance.ask_grok.return_value = {
        "intentie": ai_intent_grok, "confidence": ai_conf_grok, "reflectie": "Grok Test"
    }

    mock_prompt_builder_instance = MockPromptBuilder.return_value
    mock_prompt_builder_instance.generate_prompt_with_data.return_value = "Test exit prompt"

    # Instantiate ExitOptimizer
    # Similar to EntryDecider, ensure __init__ dependencies are mocked if complex.
    # ExitOptimizer initializes: ParamsManager, PromptBuilder, GPTReflector, GrokReflector, CNNPatterns.
    with patch('core.exit_optimizer.ParamsManager', return_value=mock_params_manager_instance), \
         patch('core.exit_optimizer.PromptBuilder', return_value=mock_prompt_builder_instance), \
         patch('core.exit_optimizer.GPTReflector', return_value=mock_gpt_reflector_instance), \
         patch('core.exit_optimizer.GrokReflector', return_value=mock_grok_reflector_instance), \
         patch('core.exit_optimizer.CNNPatterns', return_value=mock_cnn_patterns_instance):
        exit_optimizer = ExitOptimizer()

    # Prepare inputs for should_exit
    symbol = "ETH/USDT"
    current_strategy_id = "TestExitStrategy"

    mock_trade = {
        "id": "test_trade_123", "pair": symbol, "strategy": current_strategy_id,
        "profit_pct": trade_profit_pct, "open_date": datetime.utcnow() - timedelta(hours=5),
        "stake_amount": 1000, "current_profit": trade_profit_pct * 1000,
    }

    # Ensure dataframe for current_timeframe matches the one in mock_cnn_output_key (e.g., '5m')
    current_timeframe_for_test = mock_cnn_output_key.split('_')[0] # Extracts '5m' from '5m_no_bullFlag_score'
    mock_df_current = pd.DataFrame({
        'open': np.random.rand(50), 'high': np.random.rand(50), 'low': np.random.rand(50),
        'close': np.random.rand(50), 'volume': np.random.rand(50)
    })
    mock_df_current.attrs['timeframe'] = current_timeframe_for_test

    candles_by_timeframe = {current_timeframe_for_test: mock_df_current}

    decision = await exit_optimizer.should_exit(
        dataframe=mock_df_current, trade=mock_trade, symbol=symbol,
        current_strategy_id=current_strategy_id, candles_by_timeframe=candles_by_timeframe
    )

    # Assertions
    assert decision['exit'] == expected_exit, f"Scenario '{scenario_name}': Exit decision mismatch. Reason: {decision.get('reason')}"
    if expected_reason_part:
        assert expected_reason_part in decision['reason'], f"Scenario '{scenario_name}': Expected reason part '{expected_reason_part}' not in actual reason '{decision['reason']}'"

    # Verify specific get_param calls
    mock_params_manager_instance.get_param.assert_any_call("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
    mock_params_manager_instance.get_param.assert_any_call("strongPatternThreshold", strategy_id=current_strategy_id, default=0.5)

    if expected_exit and expected_reason_part == "bearish_pattern_with_ai_confirmation":
        mock_params_manager_instance.get_param.assert_any_call("exitPatternConfThreshold", strategy_id=current_strategy_id, default=0.5)

    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.assert_called_once_with(candles_by_timeframe, symbol)

# Imports for ExitOptimizer tests
from core.exit_optimizer import ExitOptimizer
# Most dependencies are already imported or can be patched via core.exit_optimizer.*

# Imports for AIOptimizer tests
import sqlite3
from core.ai_optimizer import AIOptimizer, FREQTRADE_DB_PATH as AIO_FREQTRADE_DB_PATH # Import constant too
from core.strategy_manager import StrategyManager # For AIOptimizer dependency
from core.pre_trainer import PreTrainer # For AIOptimizer dependency
# reflectie_analyser functions are used by AIOptimizer, will need mocking or direct import if used.
# from core.reflectie_analyser import analyse_reflecties, generate_mutation_proposal, analyze_timeframe_bias


# Path for the dummy reflection log, matching what AIOptimizer's __main__ used
TEST_REFLECTION_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory', 'test_reflectie-logboek.json')


@pytest.fixture
def dummy_freqtrade_db_and_reflection_log(tmp_path):
    """
    Sets up a dummy Freqtrade SQLite database with test trades and a mock reflection log.
    Yields the path to the dummy database and cleans it up afterwards.
    The reflection log is also cleaned up.
    """
    # Use tmp_path for the dummy database to ensure isolation between test runs
    dummy_db_path = tmp_path / "test_freqtrade.sqlite"

    conn_test = sqlite3.connect(dummy_db_path)
    cursor = conn_test.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            pair TEXT NOT NULL,
            strategy TEXT NOT NULL,
            profit_abs REAL,
            profit_pct REAL,
            is_open INTEGER,
            open_date TIMESTAMP,
            close_date TIMESTAMP
        );
    """)
    dummy_trades_data = [
        (1, "ETH/USDT", "DUOAI_Strategy", 10.0, 0.05, 0, datetime(2024,1,1).isoformat(), datetime(2024,1,2).isoformat()),
        (2, "ETH/USDT", "DUOAI_Strategy", -5.0, -0.025, 0, datetime(2024,1,3).isoformat(), datetime(2024,1,4).isoformat()),
        (3, "ETH/USDT", "DUOAI_Strategy", 8.0, 0.04, 0, datetime(2024,1,5).isoformat(), datetime(2024,1,6).isoformat()),
        (4, "BTC/USDT", "Another_Strategy", 20.0, 0.03, 0, datetime(2024,1,1).isoformat(), datetime(2024,1,2).isoformat()),
        (5, "ZEN/USDT", "DUOAI_Strategy", 15.0, 0.06, 0, (datetime.now() - timedelta(days=10)).isoformat(), (datetime.now() - timedelta(days=9)).isoformat()),
        (6, "ZEN/USDT", "DUOAI_Strategy", 2.0, 0.01, 0, (datetime.now() - timedelta(days=5)).isoformat(), (datetime.now() - timedelta(days=4)).isoformat()),
        (8, "ZEN/USDT", "DUOAI_Strategy", 10.0, 0.05, 0, (datetime.now() - timedelta(days=12)).isoformat(), (datetime.now() - timedelta(days=11)).isoformat()),
        (9, "ZEN/USDT", "DUOAI_Strategy", 12.0, 0.04, 0, (datetime.now() - timedelta(days=8)).isoformat(), (datetime.now() - timedelta(days=7)).isoformat()),
        (10, "ZEN/USDT", "DUOAI_Strategy", 9.0, 0.03, 0, (datetime.now() - timedelta(days=3)).isoformat(), (datetime.now() - timedelta(days=2)).isoformat()),
        (7, "LSK/BTC", "DUOAI_Strategy", 5.0, 0.02, 0, (datetime.now() - timedelta(days=2)).isoformat(), (datetime.now() - timedelta(days=1)).isoformat()),
        (11, "ADA/USDT", "DUOAI_Strategy", 5.0, 0.010, 0, (datetime.now() - timedelta(days=15)).isoformat(), (datetime.now() - timedelta(days=14)).isoformat()),
        (12, "ADA/USDT", "DUOAI_Strategy", 6.0, 0.012, 0, (datetime.now() - timedelta(days=13)).isoformat(), (datetime.now() - timedelta(days=12)).isoformat()),
        (13, "ADA/USDT", "DUOAI_Strategy", -2.0, -0.004, 0, (datetime.now() - timedelta(days=11)).isoformat(), (datetime.now() - timedelta(days=10)).isoformat()),
        (14, "ADA/USDT", "DUOAI_Strategy", 8.0, 0.015, 0, (datetime.now() - timedelta(days=9)).isoformat(), (datetime.now() - timedelta(days=8)).isoformat()),
        (15, "ADA/USDT", "DUOAI_Strategy", 7.0, 0.013, 0, (datetime.now() - timedelta(days=7)).isoformat(), (datetime.now() - timedelta(days=6)).isoformat())
    ]
    cursor.executemany("INSERT OR IGNORE INTO trades (id, pair, strategy, profit_abs, profit_pct, is_open, open_date, close_date) VALUES (?,?,?,?,?,?,?,?)", dummy_trades_data)
    conn_test.commit()
    conn_test.close()

    # Mock reflectie logboek
    mock_log_data = [
        {"token": "ETH/USDT", "strategyId": "DUOAI_Strategy", "combined_confidence": 0.8, "combined_bias_reported": 0.7,
         "current_learned_bias": 0.6, "current_learned_confidence": 0.7,
         "trade_context": {"timeframe": "1h", "profit_pct": 0.02}, "timestamp": "2025-06-11T10:00:00Z"},
    ]
    os.makedirs(os.path.dirname(TEST_REFLECTION_LOG_FILE), exist_ok=True)
    with open(TEST_REFLECTION_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(mock_log_data, f, indent=2)

    # Patch the FREQTRADE_DB_PATH constant in ai_optimizer module to use the dummy DB
    with patch('core.ai_optimizer.FREQTRADE_DB_PATH', str(dummy_db_path)):
        yield str(dummy_db_path), TEST_REFLECTION_LOG_FILE # Provide paths to the test

    # Cleanup: The dummy_db_path in tmp_path is auto-cleaned by pytest.
    # We need to explicitly remove the reflection log file if it's not in tmp_path.
    if os.path.exists(TEST_REFLECTION_LOG_FILE):
        os.remove(TEST_REFLECTION_LOG_FILE)


class TestAIOptimizer:
    @pytest.mark.asyncio
    async def test_ai_optimizer_periodic_optimization(self, dummy_freqtrade_db_and_reflection_log, caplog):
        caplog.set_level(logging.INFO)
        dummy_db_path, mock_reflection_log_path = dummy_freqtrade_db_and_reflection_log

        # Mock dependencies of AIOptimizer.
        # PreTrainer and StrategyManager are instantiated in AIOptimizer's __init__.
        # reflectie_analyser functions are called by run_periodic_optimization.

        mock_pre_trainer_instance = MagicMock(spec=PreTrainer)
        mock_pre_trainer_instance.run_pretraining_pipeline = AsyncMock()

        mock_strategy_manager_instance = MagicMock(spec=StrategyManager)
        mock_strategy_manager_instance.get_strategy_performance = MagicMock(return_value={"trades": 10, "profit_sum_pct": 0.5})
        mock_strategy_manager_instance.mutate_strategy = AsyncMock(return_value=True)
        # Mock ParamsManager instance that is an attribute of StrategyManager
        mock_params_manager_for_strat_manager = MagicMock(spec=ParamsManager)
        mock_params_manager_for_strat_manager.get_param = AsyncMock(side_effect=lambda key, strategy_id=None, default=None: default if default is not None else (5 if key == "preferredPairsCount" else ({} if key == "strategies" else [])))
        mock_params_manager_for_strat_manager.set_param = AsyncMock()
        mock_strategy_manager_instance.params_manager = mock_params_manager_for_strat_manager


        # Mock the external analysis functions
        mock_analyse_reflecties = MagicMock(return_value={"insight": "test_insight"})
        mock_generate_mutation_proposal = MagicMock(return_value={"param_change": "test_change"})
        mock_analyze_timeframe_bias = MagicMock(return_value=0.55) # Example bias score

        with patch('core.ai_optimizer.PreTrainer', return_value=mock_pre_trainer_instance) as mock_pre_trainer_class, \
             patch('core.ai_optimizer.StrategyManager', return_value=mock_strategy_manager_instance) as mock_strategy_manager_class, \
             patch('core.ai_optimizer.analyse_reflecties', mock_analyse_reflecties) as mock_analyse_reflecties_func, \
             patch('core.ai_optimizer.generate_mutation_proposal', mock_generate_mutation_proposal) as mock_generate_mutation_proposal_func, \
             patch('core.ai_optimizer.analyze_timeframe_bias', mock_analyze_timeframe_bias) as mock_analyze_timeframe_bias_func, \
             patch('core.reflectie_analyser.REFLECTION_LOG_FILE', mock_reflection_log_path): # Ensure analyse_reflecties uses the mock log

            optimizer = AIOptimizer() # Now uses mocked PreTrainer and StrategyManager

            test_symbols = ["ETH/USDT", "ZEN/USDT", "LSK/BTC", "ADA/USDT"]
            test_timeframes = ["1h"]

            logging.info("--- Test AIOptimizer: run_periodic_optimization (Default preferredPairsCount=5) ---")
            # Ensure default is used first by resetting get_param side_effect for preferredPairsCount specifically
            async def params_get_side_effect_default(key, strategy_id=None, default=None):
                if key == "preferredPairsCount": return 5 # Default for this run
                if key == "strategies": return {}
                if key == "preferredPairs": return []
                return default
            mock_params_manager_for_strat_manager.get_param.side_effect = params_get_side_effect_default

            await optimizer.run_periodic_optimization(test_symbols, test_timeframes)

            # Assertions for default run
            # Check preferredPairs after optimization
            # The actual call to set_param for preferredPairs is what we need to check
            set_param_calls = mock_params_manager_for_strat_manager.set_param.call_args_list
            preferred_pairs_call = next((c for c in set_param_calls if c.args[0] == "preferredPairs"), None)
            assert preferred_pairs_call is not None, "set_param was not called for 'preferredPairs'"

            preferred_pairs_after_opt_default = preferred_pairs_call.args[1]
            logging.info(f"Geleerde preferredPairs na optimalisatie (Default Count): {preferred_pairs_after_opt_default}")
            assert len(preferred_pairs_after_opt_default) <= 5
            assert "ZEN/USDT" in preferred_pairs_after_opt_default
            assert "ADA/USDT" in preferred_pairs_after_opt_default
            # Verify that mutation proposal was called for each symbol/timeframe pair
            assert mock_generate_mutation_proposal_func.call_count == len(test_symbols) * len(test_timeframes)
            # Verify strategy was mutated (or attempted)
            assert mock_strategy_manager_instance.mutate_strategy.call_count == len(test_symbols) * len(test_timeframes)


            logging.info("--- Test AIOptimizer: run_periodic_optimization (Custom preferredPairsCount=1) ---")
            # Reset mocks for the second run if necessary (e.g., call counts)
            mock_params_manager_for_strat_manager.reset_mock() # Resets set_param calls and get_param side_effect
            mock_generate_mutation_proposal_func.reset_mock()
            mock_strategy_manager_instance.mutate_strategy.reset_mock()

            async def params_get_side_effect_custom(key, strategy_id=None, default=None):
                if key == "preferredPairsCount": return 1 # Custom for this run
                if key == "strategies": return {}
                if key == "preferredPairs": return []
                return default
            mock_params_manager_for_strat_manager.get_param.side_effect = params_get_side_effect_custom

            await optimizer.run_periodic_optimization(test_symbols, test_timeframes)

            # Assertions for custom run
            set_param_calls_custom = mock_params_manager_for_strat_manager.set_param.call_args_list
            preferred_pairs_call_custom = next((c for c in set_param_calls_custom if c.args[0] == "preferredPairs"), None)
            assert preferred_pairs_call_custom is not None, "set_param was not called for 'preferredPairs' in custom run"

            preferred_pairs_after_opt_custom = preferred_pairs_call_custom.args[1]
            logging.info(f"Geleerde preferredPairs na optimalisatie (Custom Count=1): {preferred_pairs_after_opt_custom}")
            assert len(preferred_pairs_after_opt_custom) == 1
            assert "ZEN/USDT" in preferred_pairs_after_opt_custom

            assert "Periodic optimization cycle finished." in caplog.text


def get_exit_optimizer_param_side_effect(scenario_params):
    """
    Returns a side_effect function for mock_params_manager.get_param
    for ExitOptimizer tests.
    """
    def side_effect(key, strategy_id=None, default=None):
        # ParamsManager.get_param(key, strategy_id=current_strategy_id, default=...)
        val = scenario_params.get(key, default)
        # print(f"ExitParamsManager.get_param(key='{key}', strategy_id='{strategy_id}') -> returning: {val}, default was: {default}")
        return val
    return side_effect

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_name, mock_params, mock_cnn_output, ai_intent_gpt, ai_intent_grok, ai_conf_gpt, ai_conf_grok, trade_profit_pct, expected_exit, expected_reason_part",
    [
        # Scenario 1: Exit Approved - Strong Bearish Pattern Score Triggers Exit
        (
            "Exit_StrongBearishPattern_AIConfirm",
            { # mock_params for ParamsManager
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.0, # Focus on CNN score for this sub-scenario
                "strongPatternThreshold": 0.7,
                "exitPatternConfThreshold": 0.5, # AI confidence threshold for pattern-based exit
                # Ensure other exit paths are not triggered:
                "exitConvictionDropTrigger": 0.3, # AI conf (0.55) > this, so no low_ai_confidence_profit_taking
                "minProfitForLowConfExit": 0.005, # Profit is high (0.10), but AI conf is not low enough
                "exitAISellIntentConfThreshold": 0.6, # AI intent is HOLD, so this path not taken
            },
            # CNN: 5m_no_bullFlag_score: 0.9. Weighted: 0.9 * 0.8 = 0.72. >= strongPatternThreshold (0.7) -> True
            {"cnn_predictions": {"5m_no_bullFlag_score": 0.9}, "patterns": {}},
            "HOLD", "HOLD", # ai_intent_gpt, ai_intent_grok -> ai_exit_intent = False
            0.55, 0.55,     # ai_conf_gpt, ai_conf_grok -> combined_confidence = 0.55. (0.55 > exitPatternConfThreshold (0.5))
            0.10,           # trade_profit_pct (profitable)
            True,           # expected_exit
            "bearish_pattern_with_ai_confirmation", # expected_reason_part
        ),
        # Scenario 2: No Exit - Weighted Bearish Score Below strongPatternThreshold
        (
            "NoExit_WeakBearishPattern",
            { # mock_params for ParamsManager
                "cnnPatternWeight": 0.8,
                "exitRulePatternScore": 0.0,
                "strongPatternThreshold": 0.7,
                "exitPatternConfThreshold": 0.5,
                "exitConvictionDropTrigger": 0.3,
                "minProfitForLowConfExit": 0.005,
                "exitAISellIntentConfThreshold": 0.6,
            },
            # CNN: 5m_no_bullFlag_score: 0.4. Weighted: 0.4 * 0.8 = 0.32. < strongPatternThreshold (0.7) -> False
            {"cnn_predictions": {"5m_no_bullFlag_score": 0.4}, "patterns": {}},
            "HOLD", "HOLD", # ai_intent_gpt, ai_intent_grok -> ai_exit_intent = False
            0.8, 0.8,       # ai_conf_gpt, ai_conf_grok -> combined_confidence = 0.8 (high, but doesn't matter for this path)
            0.10,           # trade_profit_pct
            False,          # expected_exit
            "no_ai_exit_signal", # expected_reason_part (because no other condition met)
        ),
    ]
)
@patch('core.exit_optimizer.GPTReflector')
@patch('core.exit_optimizer.GrokReflector')
@patch('core.exit_optimizer.PromptBuilder')
# BiasReflector and ConfidenceEngine are not directly used by should_exit, but initialized. Mock if init fails.
@patch('core.exit_optimizer.ParamsManager')
@patch('core.exit_optimizer.CNNPatterns')
async def test_exit_optimizer_should_exit(
    MockCNNPatterns, MockParamsManager, MockPromptBuilder, MockGrokReflector, MockGPTReflector,
    scenario_name, mock_params, mock_cnn_output, ai_intent_gpt, ai_intent_grok,
    ai_conf_gpt, ai_conf_grok, trade_profit_pct, expected_exit, expected_reason_part
):
    # Setup mocks for instances
    mock_params_manager_instance = MockParamsManager.return_value
    mock_params_manager_instance.get_param.side_effect = get_exit_optimizer_param_side_effect(mock_params)

    mock_cnn_patterns_instance = MockCNNPatterns.return_value
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.return_value = mock_cnn_output

    mock_gpt_reflector_instance = MockGPTReflector.return_value
    mock_gpt_reflector_instance.ask_ai.return_value = {
        "intentie": ai_intent_gpt, "confidence": ai_conf_gpt, "reflectie": "GPT Test"
    }

    mock_grok_reflector_instance = MockGrokReflector.return_value
    mock_grok_reflector_instance.ask_grok.return_value = {
        "intentie": ai_intent_grok, "confidence": ai_conf_grok, "reflectie": "Grok Test"
    }

    mock_prompt_builder_instance = MockPromptBuilder.return_value
    mock_prompt_builder_instance.generate_prompt_with_data.return_value = "Test exit prompt"

    # Instantiate ExitOptimizer
    exit_optimizer = ExitOptimizer()

    # Prepare inputs for should_exit
    symbol = "ETH/USDT"
    current_strategy_id = "TestExitStrategy"

    mock_trade = {
        "id": "test_trade_123",
        "symbol": symbol, # Should be pair
        "pair": symbol,
        "strategy": current_strategy_id,
        "profit_pct": trade_profit_pct,
        "open_date": datetime.utcnow() - timedelta(hours=5),
        "stake_amount": 1000,
        "current_profit": trade_profit_pct * 1000, # Simplified
    }

    mock_df_5m = pd.DataFrame({
        'open': np.random.rand(50), 'high': np.random.rand(50), 'low': np.random.rand(50),
        'close': np.random.rand(50), 'volume': np.random.rand(50)
    })
    mock_df_5m.attrs['timeframe'] = '5m' # For CNN key construction like 5m_no_bullFlag_score

    candles_by_timeframe = {'5m': mock_df_5m}

    decision = await exit_optimizer.should_exit(
        dataframe=mock_df_5m,
        trade=mock_trade,
        symbol=symbol, # Symbol is used for logging/context, pair from trade for data
        current_strategy_id=current_strategy_id,
        candles_by_timeframe=candles_by_timeframe
        # learned_bias, learned_confidence, exit_conviction_drop_trigger use defaults in should_exit
    )

    # Assertions
    assert decision['exit'] == expected_exit, f"Scenario '{scenario_name}': Exit decision mismatch. Reason: {decision.get('reason')}"
    if expected_reason_part:
        assert expected_reason_part in decision['reason'], f"Scenario '{scenario_name}': Expected reason part '{expected_reason_part}' not in actual reason '{decision['reason']}'"

    # Verify specific get_param calls
    mock_params_manager_instance.get_param.assert_any_call("cnnPatternWeight", strategy_id=current_strategy_id, default=1.0)
    mock_params_manager_instance.get_param.assert_any_call("strongPatternThreshold", strategy_id=current_strategy_id, default=0.5)

    if expected_exit and expected_reason_part == "bearish_pattern_with_ai_confirmation":
        mock_params_manager_instance.get_param.assert_any_call("exitPatternConfThreshold", strategy_id=current_strategy_id, default=0.5)

    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.assert_called_once_with(candles_by_timeframe, symbol)

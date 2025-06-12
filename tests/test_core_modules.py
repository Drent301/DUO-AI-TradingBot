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
    entry_decider = EntryDecider()
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
    if not expected_enter and "pattern_score_low" in expected_reason_part : # Only check if this path was taken
         pass # Already covered by reason string check

    if expected_enter or ("final_ai_conf_low" not in decision['reason'] and "ai_intent_not_long" not in decision['reason']):
        # Check bias threshold if other conditions didn't lead to early exit from check
        mock_params_manager_instance.get_param.assert_any_call("entryLearnedBiasThreshold", strategy_id=current_strategy_id, default=0.55)

    # Verify CNNPatterns.detect_patterns_multi_timeframe was called
    mock_cnn_patterns_instance.detect_patterns_multi_timeframe.assert_called_once_with(candles_by_timeframe, symbol)

# Imports for ExitOptimizer tests
from core.exit_optimizer import ExitOptimizer
# Most dependencies are already imported or can be patched via core.exit_optimizer.*

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

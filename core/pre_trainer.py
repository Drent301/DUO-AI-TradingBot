# core/pre_trainer.py
import logging
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio # Nodig voor de async main-test
import talib.abstract as ta # For indicator calculation in prepare_training_data
import dotenv # Added for __main__

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split # Ensure this is present

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'memory')
PRETRAIN_LOG_FILE = os.path.join(MEMORY_DIR, 'pre_train_log.json')
TIME_EFFECTIVENESS_FILE = os.path.join(MEMORY_DIR, 'time_effectiveness.json')
# Use 'data/models' as specified for PyTorch model
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'models')
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_patterns_model.pth')
CNN_SCALER_PARAMS_PATH = os.path.join(MODELS_DIR, 'cnn_scaler_params.json') # Path for scaler parameters

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure MODELS_DIR is created (especially data/models)

# Define SimpleCNN class
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, sequence_length=30): # Added sequence_length, num_classes
        super(SimpleCNN, self).__init__()
        self.sequence_length = sequence_length # Store sequence_length
        self.input_channels = input_channels # Store input_channels
        self.num_classes = num_classes # Store num_classes

        self.conv1 = nn.Conv1d(in_channels=self.input_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the flattened size for the linear layer using a dummy input
        # Create a dummy input tensor with the specified sequence_length and input_channels
        # Shape: (batch_size, input_channels, sequence_length) - batch_size can be 1 for this purpose
        dummy_input = torch.randn(1, self.input_channels, self.sequence_length)

        # Pass the dummy input through the convolutional and pooling layers
        x_dummy = self.pool1(self.relu1(self.conv1(dummy_input)))
        x_dummy = self.pool2(self.relu2(self.conv2(x_dummy)))

        # Calculate the number of features after flattening
        # x_dummy.shape will be (1, out_channels_conv2, sequence_length_after_pooling)
        self.flatten_size = x_dummy.shape[1] * x_dummy.shape[2]

        # Define the fully connected layer
        self.fc = nn.Linear(self.flatten_size, self.num_classes) # Renamed fc1 to fc for clarity

    def forward(self, x):
        # Pass input through conv and pool layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Flatten the output from conv/pool layers
        # The view shape is (batch_size, flatten_size)
        x = x.view(-1, self.flatten_size)

        # Pass through the fully connected layer
        x = self.fc(x)
        return x

# Removed _set_num_classes_for_fc_init as num_classes is now handled in __init__


class PreTrainer:
    """
    Module voor pre-learning op historische Binance data.
    Verwerkt indicatoren, CNN-labels en analyseert tijd-van-dag effectiviteit.
    Nu met PyTorch CNN training.
    """

    def __init__(self, params_manager=None): # Added params_manager
        if params_manager is None:
            from core.params_manager import ParamsManager # Import here to avoid circular dependency
            self.params_manager = ParamsManager()
            logger.info("ParamsManager niet meegegeven, standaard initialisatie gebruikt in PreTrainer.")
        else:
            self.params_manager = params_manager
        logger.info("PreTrainer geïnitialiseerd.")

    async def fetch_historical_data(self, symbol: str, timeframe: str, limit: int = None) -> pd.DataFrame:
        if limit is None: # Fetch limit from params_manager if not provided
            limit = self.params_manager.get_param('data_fetch_limit', default=2000)
        """
        Haalt historische data op, bijvoorbeeld via Freqtrade's DataProvider
        of direct via CCXT (wat in bitvavo_executor zit).
        Voor pre-training, gebruiken we hier een gesimuleerde fetch.
        """
        logger.info(f"Simuleren ophalen historische data voor {symbol} ({timeframe})...")
        # In een echte implementatie:
        # from freqtrade.data.dataprovider import DataProvider
        # from freqtrade.configuration import Configuration
        # config = Configuration.from_files([]) # Minimal config
        # dp = DataProvider(config, None, None)
        # candles_df = dp.get_pair_dataframe(pair=symbol, timeframe=timeframe, candle_type='mark') # or 'spot'
        # if candles_df is None or candles_df.empty:
        #     logger.error(f"Kon geen historische data ophalen voor {symbol} ({timeframe}) via DataProvider.")
        #     return pd.DataFrame()
        # return candles_df.tail(limit)


        # Dummy data generation
        data = []
        now = datetime.utcnow() # Use UTC for consistency
        interval_seconds_map = {
            '1m': 60, '5m': 300, '15m': 900, '1h': 3600,
            '4h': 14400, '12h': 43200, '1d': 86400, '1w': 604800
        }
        interval_seconds = interval_seconds_map.get(timeframe, 300)

        for i in range(limit):
            date = now - timedelta(seconds=(limit - 1 - i) * interval_seconds)
            open_ = 100 + i * 0.01 * (1 + np.random.randn() * 0.01) # More realistic price progression
            close_ = open_ + (np.random.randn() * 0.5)
            high_ = max(open_, close_) + (np.random.rand() * 0.2)
            low_ = min(open_, close_) - (np.random.rand() * 0.2)
            volume = 1000 + np.random.rand() * 5000 # Wider volume range
            data.append([date, open_, high_, low_, close_, volume])

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.attrs['timeframe'] = timeframe
        df.attrs['pair'] = symbol # Store pair as an attribute
        logger.info(f"Historische data gesimuleerd voor {symbol} ({timeframe}): {len(df)} candles.")
        return df

    async def prepare_training_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Verwerkt een dataframe door indicatoren toe te voegen en CNN-labels te genereren.
        """
        pair = dataframe.attrs.get('pair', 'N/A')
        tf = dataframe.attrs.get('timeframe', 'N/A')
        logger.info(f"Voorbereiden trainingsdata voor {pair} ({tf})...")

        if dataframe.empty:
            logger.warning("Lege dataframe ontvangen in prepare_training_data.")
            return dataframe

        # Voeg Freqtrade-compatibele indicatoren toe
        if 'rsi' not in dataframe.columns:
            dataframe['rsi'] = ta.RSI(dataframe) # Default timeperiod=14
        if 'macd' not in dataframe.columns: # Ensure all MACD components are added if one is missing
            macd_df = ta.MACD(dataframe) # Default fast=12, slow=26, signal=9
            dataframe['macd'] = macd_df['macd']
            dataframe['macdsignal'] = macd_df['macdsignal']
            dataframe['macdhist'] = macd_df['macdhist']

        # Voorbeeld: Bollinger Bands (vaak gebruikt in Freqtrade)
        if 'bb_middleband' not in dataframe.columns:
            from freqtrade.vendor.qtpylib.indicators import bollinger_bands # Local import for clarity
            bollinger = bollinger_bands(ta.TYPPRICE(dataframe), window=20, stds=2)
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_upperband'] = bollinger['upper']


        # Nieuwe bullFlag_label logica
        future_N_candles = self.params_manager.get_param('future_N_candles_for_label', default=20) # Kijk N candles vooruit
        profit_threshold_pct = self.params_manager.get_param('profit_threshold_for_label', default=0.02) # 2% winst
        loss_threshold_pct = self.params_manager.get_param('loss_threshold_for_label', default=-0.01) # 1% verlies (negatief)

        dataframe['future_high'] = dataframe['high'].shift(-future_N_candles)
        dataframe['future_low'] = dataframe['low'].shift(-future_N_candles) # Nodig voor stop-loss check
        dataframe['future_close'] = dataframe['close'].shift(-future_N_candles)

        # Bereken potentieel toekomstig rendement
        dataframe['future_profit_pct'] = (dataframe['future_high'] - dataframe['close']) / dataframe['close']

        # Labeling conditions
        # Conditie 1: Winstdoel bereikt - de toekomstige high is minstens X% boven de huidige sluitprijs
        profit_target_met = dataframe['future_profit_pct'] >= profit_threshold_pct

        # Conditie 2: Verliesdrempel geraakt - de toekomstige low is minstens Y% onder de huidige sluitprijs
        # We kijken of de low onder de stop-loss is gekomen VOORDAT de profit target eventueel is gehaald.
        # Dit is een vereenvoudiging; in werkelijkheid zou je de volgorde van H/L binnen de N candles moeten weten.
        # Voor nu: als de future_low onder de loss_threshold is, markeren we het als potentieel verlies.
        loss_threshold_met = (dataframe['future_low'] - dataframe['close']) / dataframe['close'] <= loss_threshold_pct

        # Labeling: 1 als winstdoel is bereikt EN verliesdrempel NIET is geraakt (of later is geraakt dan winst)
        # Dit is een simpele benadering. Een meer geavanceerde label zou de volgorde van events moeten overwegen.
        dataframe['bullFlag_label'] = np.where(profit_target_met & ~loss_threshold_met, 1, 0)

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist']
        cols_to_drop_for_na = ['bullFlag_label', 'future_high', 'future_close', 'future_profit_pct', 'future_low'] + feature_columns
        dataframe.dropna(subset=cols_to_drop_for_na, inplace=True)

        # Drop temporary columns used for labeling
        dataframe.drop(columns=['future_high', 'future_close', 'future_profit_pct', 'future_low'], inplace=True, errors='ignore')

        logger.info(f"Trainingsdata voorbereid met {len(dataframe)} samples na NA drop. Features: {feature_columns}")
        return dataframe


    async def train_ai_models(self, training_data: pd.DataFrame, model_type: str = 'SimpleCNN_Torch'):
        """
        Traint een SimpleCNN model met PyTorch op de voorbereide data.
        """
        logger.info(f"Start training van PyTorch AI-model '{model_type}' met {len(training_data)} initiële samples...")

        if training_data.empty:
            logger.warning(f"Geen trainingsdata voor model '{model_type}'. Training overgeslagen.")
            return

        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macdsignal', 'macdhist']
        if not all(col in training_data.columns for col in feature_columns + ['bullFlag_label']):
            logger.error(f"Benodigde kolommen ({feature_columns + ['bullFlag_label']}) niet allemaal aanwezig in training_data. Overslaan.")
            return

        sequence_length = self.params_manager.get_param('sequence_length_cnn', default=30)
        num_epochs = self.params_manager.get_param('num_epochs_cnn', default=10)
        batch_size = self.params_manager.get_param('batch_size_cnn', default=32)
        learning_rate = self.params_manager.get_param('learning_rate_cnn', default=0.001)

        # Data Preparation: Sequences and Normalization
        X_list, y_list = [], []
        for i in range(len(training_data) - sequence_length):
            X_list.append(training_data[feature_columns].iloc[i:i + sequence_length].values)
            y_list.append(training_data['bullFlag_label'].iloc[i + sequence_length - 1]) # Label from the last candle in sequence

        if not X_list:
            logger.warning(f"Niet genoeg data ({len(training_data)} samples) om sequenties te maken met lengte {sequence_length}. Training overgeslagen.")
            return

        X = np.array(X_list)
        y = np.array(y_list)

        # Normalization (feature-wise min-max scaling to [0,1])
        # Reshape X to 2D for scaling: (num_samples * sequence_length, num_features)
        X_reshaped = X.reshape(-1, X.shape[-1])
        min_vals = X_reshaped.min(axis=0)
        max_vals = X_reshaped.max(axis=0)
        # Avoid division by zero if a feature is constant
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        X_normalized_reshaped = (X_reshaped - min_vals) / range_vals
        # Reshape back to 3D: (num_samples, sequence_length, num_features)
        X_normalized = X_normalized_reshaped.reshape(X.shape)

        X_tensor = torch.tensor(X_normalized, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Train-Validation Split
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor if np.sum(y)>1 else None) # stratify if possible

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model, Criterion, Optimizer
        input_channels = X_train.shape[1] # Number of features
        # sequence_length is already defined and fetched from params_manager
        model = SimpleCNN(input_channels=input_channels, num_classes=2, sequence_length=sequence_length)
        # model._set_num_classes_for_fc_init(2) # No longer needed

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_val_precision = 0.0
        best_val_recall = 0.0
        best_val_f1 = 0.0

        logger.info(f"Start trainingsloop voor {num_epochs} epochs...")
        # Initialize best_val_accuracy before the epoch loop, alongside best_val_loss
        # best_val_loss is already initialized, best_val_accuracy was also initialized
        # No change needed here for initialization based on current code structure.
        # best_val_precision, best_val_recall, best_val_f1 are also already initialized.

        for epoch in range(num_epochs):
            model.train()
            # Initialize epoch_train_loss = 0.0 before the training loop for each epoch.
            epoch_train_loss = 0.0
            for batch_idx, (data, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # Inside the training batch loop, accumulate loss: epoch_train_loss += loss.item() * data.size(0).
                epoch_train_loss += loss.item() * data.size(0)

            # After the training batch loop, calculate average training loss:
            # avg_epoch_train_loss = epoch_train_loss / len(dataloader.dataset).
            # Using len(train_dataset) as requested if dataloader.dataset is not correct.
            # len(train_dataset) is indeed correct here.
            avg_epoch_train_loss = epoch_train_loss / len(train_dataset)

            # Validation phase
            model.eval()
            current_epoch_val_loss = 0.0 # Using a new variable for epoch-specific validation loss sum
            all_preds, all_targets = [], []
            with torch.no_grad():
                for data, targets in val_loader:
                    outputs = model(data)
                    # batch_val_loss to avoid confusion with training loss
                    batch_val_loss = criterion(outputs, targets)
                    # Accumulate validation loss: val_loss += batch_val_loss.item() * data.size(0)
                    current_epoch_val_loss += batch_val_loss.item() * data.size(0)

                    # Convert outputs to probabilities for class 1, then to predictions
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predicted_labels = (probs > 0.5).long()

                    all_preds.extend(predicted_labels.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            # Calculate average validation loss for the epoch using len(val_dataset)
            avg_epoch_val_loss = current_epoch_val_loss / len(val_dataset)
            val_accuracy = accuracy_score(all_targets, all_preds)
            val_precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='binary', zero_division=0)
            val_f1 = f1_score(all_targets, all_preds, average='binary', zero_division=0)

            # Update the logging statement to use avg_epoch_train_loss.
            logger.debug(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_train_loss:.4f}, "
                f"Validation Loss: {avg_epoch_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, "
                f"Validation F1: {val_f1:.4f} for {model_type}"
            )

            # When val_loss < best_val_loss, also update best_val_accuracy = val_accuracy.
            if avg_epoch_val_loss < best_val_loss:
                best_val_loss = avg_epoch_val_loss
                best_val_accuracy = val_accuracy # This was already here
                best_val_precision = val_precision # Keep updating other metrics too
                best_val_recall = val_recall # Keep updating other metrics too
                best_val_f1 = val_f1 # Keep updating other metrics too
                torch.save(model.state_dict(), CNN_MODEL_PATH)
                logger.debug(f"Model voor {model_type} opgeslagen. Val Loss: {best_val_loss:.4f}, Acc: {best_val_accuracy:.4f}, Prec: {best_val_precision:.4f}, Rec: {best_val_recall:.4f}, F1: {best_val_f1:.4f}")

        # At the end of the training for a pattern (after the epoch loop),
        # ensure that final_val_loss is best_val_loss and final_val_accuracy is best_val_accuracy
        # before calling _log_pretrain_activity.
        # The _log_pretrain_activity call already uses best_val_loss, best_val_accuracy, etc.
        # So no explicit assignment like val_loss = best_val_loss is needed here if those are used.

        # Save Model (this happens if last epoch was best, or to save final state regardless)
        # The current logic saves only if val_loss improves. This is generally what we want.
        try:
            # Ensure the best model is what's saved, which is handled by the check above.
            # Check if a model was actually saved (i.e., if best_val_loss was updated from inf)
            if os.path.exists(CNN_MODEL_PATH): # Check if the model file was created/updated
                 logger.info(f"PyTorch model '{model_type}' training voltooid. Beste model opgeslagen op {CNN_MODEL_PATH}")
            else:
                 # This case could happen if no improvement was seen from the initial best_val_loss = float('inf')
                 # or if there was an issue saving the file, though torch.save would likely error.
                 # If training_data was very small, X_list could be empty, skipping training loop.
                 # In such cases, best_val_loss would remain 'inf'.
                 logger.warning(f"PyTorch model '{model_type}' training voltooid, maar er is mogelijk geen model opgeslagen (of het is niet verbeterd). Controleer logs.")

            # Save scaler parameters
            scaler_params = {
                'feature_columns': feature_columns,
                'min_vals': min_vals.tolist(),
                'max_vals': max_vals.tolist(),
                'sequence_length': sequence_length # Add sequence_length to scaler_params
            }
            with open(CNN_SCALER_PARAMS_PATH, 'w') as f:
                json.dump(scaler_params, f, indent=4)
            logger.info(f"CNN scaler parameters opgeslagen op {CNN_SCALER_PARAMS_PATH}")

        except Exception as e:
            logger.error(f"Fout bij opslaan PyTorch model of scaler parameters: {e}")
            return

        # Log pre-training activiteit
        await asyncio.to_thread(
            self._log_pretrain_activity,
            model_type=model_type, # Use the passed model_type
            data_size=len(X_train) + len(X_val), # Total sequences used
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            best_val_precision=best_val_precision,
            best_val_recall=best_val_recall,
            best_val_f1=best_val_f1
        )
        logger.info(f"Pre-training voltooid voor {model_type}. Beste validatieverlies: {best_val_loss:.4f}, Beste nauwkeurigheid: {best_val_accuracy:.4f}")
        # Return best_val_loss or a dict of all best metrics if needed by caller
        return best_val_loss


    def _log_pretrain_activity(self, model_type: str, data_size: int, best_val_loss: float = None, best_val_accuracy: float = None, best_val_precision: float = None, best_val_recall: float = None, best_val_f1: float = None):
        """ Logt de pre-trainingsactiviteit. """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "data_size": data_size,
            "status": "completed_pytorch_training",
            "model_path_saved": CNN_MODEL_PATH,
            "scaler_params_path_saved": CNN_SCALER_PARAMS_PATH,
            "best_validation_loss": best_val_loss,
            "best_validation_accuracy": best_val_accuracy,
            "best_validation_precision": best_val_precision,
            "best_validation_recall": best_val_recall,
            "best_validation_f1": best_val_f1
        }
        logs = []
        try:
            if os.path.exists(PRETRAIN_LOG_FILE) and os.path.getsize(PRETRAIN_LOG_FILE) > 0: # Check size
                try:
                    with open(PRETRAIN_LOG_FILE, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content: # Ensure content is not empty
                            logs = json.loads(content)
                            if not isinstance(logs, list): logs = [logs] # Handle single object case
                        else: # File is empty
                            logs = []
                except json.JSONDecodeError:
                    logger.warning(f"Kon {PRETRAIN_LOG_FILE} niet parsen, start met nieuwe log.")
                    logs = [] # Reset if file is corrupt
            logs.append(entry)
            with open(PRETRAIN_LOG_FILE, 'w', encoding='utf-8') as f: # Overwrite with updated logs
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Fout bij loggen pre-train activiteit: {e}")

    async def analyze_time_of_day_effectiveness(self, historical_data: pd.DataFrame, strategy_id: str) -> Dict[str, Any]:
        """
        Analyseert de prestatie van een strategie per dagdeel.
        [cite_start]Vertaald van timeOfDayEffectiveness[cite: 71, 141].
        """
        logger.info(f"Analyseren tijd-van-dag effectiviteit voor strategie {strategy_id}...")

        if historical_data.empty:
            logger.warning("Geen historische data voor tijd-van-dag analyse.")
            return {}

        # We gebruiken 'trade_outcome_label' die al is toegevoegd in prepare_training_data
        if 'trade_outcome_label' not in historical_data.columns:
            logger.warning("Kolom 'trade_outcome_label' ontbreekt. Kan geen tijd-van-dag effectiviteit analyseren.")
            return {} # Of voeg hier een dummy toe als dat zinvol is voor de flow

        # Zorg ervoor dat de index een DatetimeIndex is
        if not isinstance(historical_data.index, pd.DatetimeIndex):
            logger.error("DataFrame index is geen DatetimeIndex. Kan uur niet extraheren.")
            return {}

        df_copy = historical_data.copy() # Werk op een kopie
        df_copy['hour_of_day'] = df_copy.index.hour

        # Bereken gemiddelde outcome per uur
        # Alleen rijen waar outcome niet 0 is (dus daadwerkelijke winst/verlies)
        time_effectiveness = df_copy[df_copy['trade_outcome_label'] != 0].groupby('hour_of_day')['trade_outcome_label'].agg(['mean', 'count']).reset_index()
        time_effectiveness.rename(columns={'mean': 'avg_outcome', 'count': 'num_trades'}, inplace=True)

        # Converteer naar dictionary voor JSON opslag
        result_dict = time_effectiveness.set_index('hour_of_day').to_dict(orient='index')

        logger.info(f"Tijd-van-dag effectiviteit geanalyseerd: {result_dict}")

        # Sla op naar bestand
        # Use await asyncio.to_thread for blocking file I/O
        def save_time_effectiveness():
            with open(TIME_EFFECTIVENESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2)
        await asyncio.to_thread(save_time_effectiveness)
        return result_dict

    async def run_pretraining_pipeline(self, symbol: str, timeframe: str, strategy_id: str, params_manager=None): # Added params_manager
        """
        Voert de volledige pre-training pipeline uit met PyTorch CNN.
        """
        logger.info(f"Start pre-training pipeline voor {symbol} ({timeframe}, strategie: {strategy_id})...")

        # Ensure self.params_manager is set, if not passed, it's initialized in __init__
        if params_manager is not None and self.params_manager != params_manager: # If passed, ensure it's used
            self.params_manager = params_manager
            logger.info("ParamsManager extern meegegeven aan run_pretraining_pipeline.")


        historical_df = await self.fetch_historical_data(symbol, timeframe) # limit is now handled by fetch_historical_data via params_manager
        if historical_df.empty:
            logger.error(f"Kan pre-training niet uitvoeren: geen historische data voor {symbol} ({timeframe}).")
            return

        processed_df = await self.prepare_training_data(historical_df)
        if processed_df.empty:
            logger.error(f"Kan pre-training niet uitvoeren: geen data na voorbereiding voor {symbol} ({timeframe}).")
            return

        # Train het CNN model
        await self.train_ai_models(processed_df, model_type='SimpleCNN_Torch')

        # Analyseer tijd-van-dag effectiviteit (conditioneel)
        # De nieuwe prepare_training_data maakt 'bullFlag_label', niet 'trade_outcome_label'.
        # Als 'analyze_time_of_day_effectiveness' nog steeds 'trade_outcome_label' vereist,
        # moet die kolom apart worden gegenereerd of de analysefunctie aangepast.
        # Voor nu, houden we de check zoals in het voorbeeld van de gebruiker.
        if 'trade_outcome_label' in processed_df.columns:
            await self.analyze_time_of_day_effectiveness(processed_df, strategy_id)
        else:
            logger.info("Kolom 'trade_outcome_label' niet gevonden in processed_df. Overslaan time_of_day_effectiveness.")

        logger.info(f"Pre-training pipeline voltooid voor {symbol} ({timeframe}).")

# Voorbeeld van hoe je het zou kunnen gebruiken (voor testen)
if __name__ == "__main__":
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if not os.path.exists(dotenv_path):
        print(f"LET OP: .env bestand niet gevonden op {dotenv_path}. Maak het aan indien nodig.")
    dotenv.load_dotenv(dotenv_path) # Load environment variables

    # Setup basic logging for the test
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    # Import ParamsManager here for the test block
    try:
        from core.params_manager import ParamsManager
    except ImportError:
        logger.error("ParamsManager kon niet worden geïmporteerd. Zorg ervoor dat het pad correct is.")
        sys.exit(1) # Exit if essential components are missing for the test

    # Dummy Freqtrade database setup (simplified, only if absolutely needed by params_manager or other components)
    # For this test, we'll assume params_manager can run without a full Freqtrade setup if defaults are used.
    # If Freqtrade's DB is strictly required by params_manager, this part would need actual Freqtrade setup.
    # db_url = os.getenv('FREQTRADE_DB_URL', 'sqlite:///freqtrade_test_pretrain.sqlite')
    # logger.info(f"Gebruik Freqtrade DB URL: {db_url} (uit .env of default)")
    # if not os.path.exists(db_url.replace('sqlite:///', '')) and 'sqlite' in db_url:
    #     logger.info(f"Test database {db_url} niet gevonden. Dit kan problemen veroorzaken als ParamsManager het vereist.")
        # In een CI/CD of dev setup, zou je hier wellicht een dummy DB willen initialiseren.


    async def run_test_pre_trainer():
        try:
            params_manager_instance = ParamsManager() # Initialize with default path or ensure config exists
            # Voorbeeld: stel een parameter in als dat nodig is voor de test
            params_manager_instance.set_param('data_fetch_limit', 500) # Kleinere dataset voor snellere test
            params_manager_instance.set_param('sequence_length_cnn', 10)
            params_manager_instance.set_param('num_epochs_cnn', 3) # Minder epochs voor snelle test
            logger.info(f"ParamsManager ingesteld voor test: data_fetch_limit={params_manager_instance.get_param('data_fetch_limit')}")
        except Exception as e:
            logger.error(f"Fout bij initialiseren/instellen ParamsManager: {e}")
            return

        pre_trainer = PreTrainer(params_manager=params_manager_instance)
        test_symbol = "ETH/USDT" # Gebruik een symbool dat je data provider kan leveren (gesimuleerd hier)
        test_timeframe = "1h"
        test_strategy_id = "TestStrategyPyTorch" # Willekeurige ID voor test

        print("\n--- Test PreTrainer Pipeline (PyTorch CNN) ---")
        await pre_trainer.run_pretraining_pipeline(test_symbol, test_timeframe, test_strategy_id, params_manager=params_manager_instance)

        print(f"\nControleer logs in {PRETRAIN_LOG_FILE} en {TIME_EFFECTIVENESS_FILE}")
        if os.path.exists(CNN_MODEL_PATH):
            print(f"SUCCES: PyTorch CNN model opgeslagen op: {CNN_MODEL_PATH}")
            print(f"Modelgrootte: {os.path.getsize(CNN_MODEL_PATH)} bytes")
        else:
            print(f"FOUT: PyTorch CNN model NIET opgeslagen op: {CNN_MODEL_PATH}")

        if os.path.exists(CNN_SCALER_PARAMS_PATH):
            print(f"SUCCES: CNN scaler parameters opgeslagen op: {CNN_SCALER_PARAMS_PATH}")
            with open(CNN_SCALER_PARAMS_PATH, 'r') as f:
                scaler_params_content = json.load(f)
            print(f"Scaler params content (eerste 5 features min/max):")
            for i, col in enumerate(scaler_params_content['feature_columns'][:5]):
                print(f"  {col}: min={scaler_params_content['min_vals'][i]:.4f}, max={scaler_params_content['max_vals'][i]:.4f}")
        else:
            print(f"FOUT: CNN scaler parameters NIET opgeslagen op: {CNN_SCALER_PARAMS_PATH}")

    asyncio.run(run_test_pre_trainer())

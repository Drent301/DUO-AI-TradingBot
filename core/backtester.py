import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio

from core.params_manager import ParamsManager
from core.cnn_patterns import CNNPatterns
from core.bitvavo_executor import BitvavoExecutor
# import talib.abstract as ta # Defer import to where it's used

# Ensure the 'memory' directory exists
log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'memory'
log_dir.mkdir(parents=True, exist_ok=True)
BACKTEST_RESULTS_FILE = log_dir / 'backtest_results.json'

logger = logging.getLogger(__name__)

# Helper for timeframe to seconds conversion
TIMEFRAME_TO_SECONDS = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '2h': 7200,
    '4h': 14400,
    '6h': 21600,
    '8h': 28800,
    '12h': 43200,
    '1d': 86400,
    '3d': 259200,
    '1w': 604800
}

class Backtester:
    def __init__(self, params_manager: ParamsManager, cnn_pattern_detector: CNNPatterns, bitvavo_executor: BitvavoExecutor):
        self.params_manager = params_manager
        self.cnn_pattern_detector = cnn_pattern_detector
        self.bitvavo_executor = bitvavo_executor
        logger.info("Backtester initialized.")

    async def _fetch_backtest_data(self, symbol: str, timeframe: str, start_date_str: str) -> pd.DataFrame:
        logger.info(f"Fetching backtest data for {symbol} ({timeframe}) from {start_date_str}")
        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_timestamp_ms = int(start_dt.timestamp() * 1000)
        except ValueError:
            logger.error(f"Invalid start_date_str format: {start_date_str}. Please use YYYY-MM-DD.")
            return pd.DataFrame()

        if not self.bitvavo_executor:
            logger.error("BitvavoExecutor is not initialized. Cannot fetch real data.")
            return pd.DataFrame()

        all_ohlcv = []
        current_since = start_timestamp_ms
        limit_per_call = 500  # Bitvavo's typical limit, adjust if necessary

        while True:
            try:
                logger.debug(f"Fetching chunk for {symbol} ({timeframe}) since {datetime.fromtimestamp(current_since / 1000, tz=timezone.utc)}")
                ohlcv_chunk = await self.bitvavo_executor.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit_per_call
                )

                if not ohlcv_chunk:
                    logger.info(f"No more data returned for {symbol} ({timeframe}) after {datetime.fromtimestamp(current_since / 1000, tz=timezone.utc)}.")
                    break

                all_ohlcv.extend(ohlcv_chunk)
                last_candle_timestamp_ms = ohlcv_chunk[-1][0]

                # Stop if the last candle is beyond the current time (or some reasonable future buffer)
                if datetime.fromtimestamp(last_candle_timestamp_ms / 1000, tz=timezone.utc) > datetime.now(timezone.utc) + timedelta(days=1):
                    logger.info(f"Fetched data up to a future point for {symbol} ({timeframe}). Stopping.")
                    # Trim data that might be too far into the future if exchange returns it
                    all_ohlcv = [c for c in all_ohlcv if c[0] <= (datetime.now(timezone.utc) + timedelta(days=1)).timestamp() * 1000]
                    break

                # If fewer candles than limit returned, it's likely the end of available data for the period
                if len(ohlcv_chunk) < limit_per_call:
                    logger.info(f"Fetched last available chunk for {symbol} ({timeframe}). Count: {len(ohlcv_chunk)}.")
                    break

                # Prepare 'since' for the next call: timestamp of the last candle + 1ms (or + timeframe_duration_ms)
                # Using last candle's timestamp + 1ms to avoid overlap but ensure continuity.
                # Some exchanges might need + timeframe_duration_ms. CCXT handles this by convention.
                current_since = last_candle_timestamp_ms + 1

                await asyncio.sleep(0.2) # Respect API rate limits

            except Exception as e:
                logger.error(f"Error fetching OHLCV data for {symbol} ({timeframe}): {e}", exc_info=True)
                return pd.DataFrame()

        if not all_ohlcv:
            logger.warning(f"No OHLCV data fetched for {symbol} ({timeframe}) from {start_date_str}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['timestamp'], inplace=True) # Ensure no overlaps if any
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        df.sort_index(inplace=True) # Ensure chronological order

        # Filter out data before the requested start_date_str, as exchanges might return earlier data
        df = df[df.index >= start_dt]

        logger.info(f"Successfully fetched {len(df)} candles for {symbol} ({timeframe}) from {start_date_str} up to {df.index[-1] if not df.empty else 'N/A'}.")
        return df

    def _add_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        pair = dataframe.attrs.get('pair', 'N/A')
        timeframe = dataframe.attrs.get('timeframe', 'N/A')
        logger.debug(f"Adding indicators for {pair} ({timeframe}) backtest data.")

        # Ensure attrs are set if not present
        if 'pair' not in dataframe.attrs:
            dataframe.attrs['pair'] = 'UnknownPair'
        if 'timeframe' not in dataframe.attrs:
            dataframe.attrs['timeframe'] = 'UnknownTimeframe'

        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(dataframe.columns):
            logger.error(f"DataFrame for {pair} ({timeframe}) is missing required columns for indicator calculation. Got: {dataframe.columns}")
            return pd.DataFrame() # Return empty if critical columns are missing

        try:
            import talib.abstract as ta
            # RSI
            dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)

            # MACD
            macd_result = ta.MACD(dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            dataframe['macd'] = macd_result['macd']
            dataframe['macdsignal'] = macd_result['macdsignal']
            dataframe['macdhist'] = macd_result['macdhist']

            # Bollinger Bands
            bb_result = ta.BBANDS(dataframe['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            dataframe['upperband'] = bb_result['upperband']
            dataframe['middleband'] = bb_result['middleband']
            dataframe['lowerband'] = bb_result['lowerband']

            dataframe.dropna(inplace=True) # Remove rows with NaN values generated by indicators
            logger.debug(f"Indicators added for {pair} ({timeframe}). Shape after adding: {dataframe.shape}")
        except ImportError:
            logger.warning(f"TA-Lib module not found in Backtester._add_indicators. Skipping TA-Lib based indicators for {pair} ({timeframe}).")
        except Exception as e:
            logger.error(f"Error adding TA-Lib indicators for {pair} ({timeframe}): {e}", exc_info=True)
            return pd.DataFrame() # Return empty on error

        return dataframe

    async def run_backtest(self, symbol: str, timeframe: str, pattern_type: str, architecture_key: str, sequence_length: int):
        logger.info(f"Starting backtest for {symbol} ({timeframe}), pattern: {pattern_type}, arch: {architecture_key}")

        global_params = self.params_manager.get_global_params()
        perform_backtesting = global_params.get('perform_backtesting', False)
        if not perform_backtesting:
            logger.info("Backtesting is disabled in parameters. Skipping.")
            return None

        backtest_start_date_str = global_params.get('backtest_start_date_str')
        entry_threshold = global_params.get('backtest_entry_threshold', 0.7)
        tp_pct = global_params.get('backtest_take_profit_pct', 0.05)
        sl_pct = global_params.get('backtest_stop_loss_pct', 0.02)
        hold_duration_candles = global_params.get('backtest_hold_duration_candles', 20)
        initial_capital = global_params.get('backtest_initial_capital', 1000.0)
        stake_pct_capital = global_params.get('backtest_stake_pct_capital', 0.1)

        if not backtest_start_date_str:
            logger.warning("`backtest_start_date_str` is not set in parameters. Skipping backtest.")
            return None

        backtest_df = await self._fetch_backtest_data(symbol, timeframe, backtest_start_date_str)
        if backtest_df.empty:
            logger.error(f"Failed to fetch data for backtest: {symbol} ({timeframe}).")
            return None

        backtest_df.attrs['pair'] = symbol
        backtest_df.attrs['timeframe'] = timeframe

        backtest_df_with_indicators = self._add_indicators(backtest_df.copy())
        if backtest_df_with_indicators.empty:
            logger.error(f"Failed to add indicators to backtest data for {symbol} ({timeframe}).")
            return None

        # Ensure CNN model is loaded for the pattern and architecture
        model_key = f"{pattern_type}_{architecture_key}_{symbol.replace('/', '_')}_{timeframe}" # Using the full key from CNNPatterns
        # Check if model is loaded, if not, CNNPatterns should attempt to load it upon predict_pattern_score.
        # However, it's good to verify if the model *can* be loaded.
        # This part assumes train_ai_models has run and saved the model.
        # We rely on predict_pattern_score to handle the actual loading via _load_cnn_models_and_scalers if needed.
        # For an explicit check, one might need a dedicated method in CNNPatterns like `is_model_available`.
        # For now, proceed and let predict_pattern_score handle it. If it fails there, it will return None.
        logger.info(f"Proceeding with backtest. Model {model_key} will be loaded by CNNPatterns on demand if not already.")


        current_capital = initial_capital
        trades = []
        open_position = None  # {'entry_price', 'asset_amount', 'entry_time_idx', 'sl_price', 'tp_price', 'hold_until_idx'}
        portfolio_history = [] # Store {'timestamp', 'capital'}

        candle_interval_seconds = TIMEFRAME_TO_SECONDS.get(timeframe)
        if not candle_interval_seconds:
            logger.error(f"Unknown timeframe for candle interval: {timeframe}. Cannot proceed.")
            return None

        logger.info(f"Backtest loop starting. Initial capital: {current_capital}. Candles: {len(backtest_df_with_indicators)}")

        for i in range(sequence_length -1, len(backtest_df_with_indicators)):
            current_candle = backtest_df_with_indicators.iloc[i]
            current_price = current_candle['close']
            current_time = backtest_df_with_indicators.index[i]

            portfolio_history.append({'timestamp': current_time, 'capital': current_capital})

            # Position Management
            if open_position:
                exit_reason = None
                # pnl = 0 # pnl is calculated at closure

                # Check Stop Loss
                if current_price <= open_position['sl_price']:
                    exit_reason = "Stop Loss"
                # Check Take Profit
                elif current_price >= open_position['tp_price']:
                    exit_reason = "Take Profit"
                # Check Hold Duration
                elif i >= open_position['hold_until_idx']:
                    exit_reason = "Hold Duration Met"

                if exit_reason:
                    exit_price = current_price
                    profit_or_loss = (exit_price - open_position['entry_price']) * open_position['asset_amount']

                    current_capital += open_position['original_stake'] + profit_or_loss

                    trades.append({
                        'entry_time': open_position['entry_time_dt'],
                        'exit_time': current_time,
                        'entry_price': open_position['entry_price'],
                        'exit_price': exit_price,
                        'asset_amount': open_position['asset_amount'],
                        'pnl': profit_or_loss,
                        'capital_after_trade': current_capital,
                        'reason_for_exit': exit_reason,
                        'pattern_score_at_entry': open_position['pattern_score_at_entry']
                    })
                    logger.debug(f"Closed position: {symbol} at {exit_price:.2f}. P&L: {profit_or_loss:.2f}. Reason: {exit_reason}. Capital: {current_capital:.2f}")
                    open_position = None

            # New Entry Check (if no position is open)
            if not open_position:
                start_idx = max(0, i - sequence_length + 1)
                end_idx = i + 1

                if end_idx - start_idx < sequence_length:
                    continue

                historical_window_df = backtest_df_with_indicators.iloc[start_idx:end_idx]

                if len(historical_window_df) < sequence_length :
                    continue

                pattern_score = self.cnn_pattern_detector.predict_pattern_score(
                    dataframe_with_indicators=historical_window_df,
                    pattern_type=pattern_type,
                    architecture_key=architecture_key,
                    symbol=symbol, # Needed for model/scaler key construction in CNNPatterns
                    timeframe=timeframe # Needed for model/scaler key construction in CNNPatterns
                )


                if pattern_score is not None and pattern_score >= entry_threshold:
                    stake_to_use = current_capital * stake_pct_capital
                    if stake_to_use <= 0 or current_capital < stake_to_use :
                        logger.warning(f"Cannot open position for {symbol}. Stake: {stake_to_use:.2f}, Capital: {current_capital:.2f}. Skipping.")
                        continue

                    asset_amount_to_buy = stake_to_use / current_price
                    current_capital -= stake_to_use

                    open_position = {
                        'entry_price': current_price,
                        'asset_amount': asset_amount_to_buy,
                        'entry_time_idx': i,
                        'entry_time_dt': current_time,
                        'sl_price': current_price * (1 - sl_pct),
                        'tp_price': current_price * (1 + tp_pct),
                        'hold_until_idx': i + hold_duration_candles,
                        'original_stake': stake_to_use,
                        'pattern_score_at_entry': pattern_score
                    }
                    logger.info(f"Opened new position: {symbol} at {current_price:.2f}. Amount: {asset_amount_to_buy:.4f}. Stake: {stake_to_use:.2f}. SL: {open_position['sl_price']:.2f}, TP: {open_position['tp_price']:.2f}. Capital after open: {current_capital:.2f}")

        if open_position:
            last_price = backtest_df_with_indicators['close'].iloc[-1]
            last_time = backtest_df_with_indicators.index[-1]
            profit_or_loss = (last_price - open_position['entry_price']) * open_position['asset_amount']
            current_capital += open_position['original_stake'] + profit_or_loss

            trades.append({
                'entry_time': open_position['entry_time_dt'],
                'exit_time': last_time,
                'entry_price': open_position['entry_price'],
                'exit_price': last_price,
                'asset_amount': open_position['asset_amount'],
                'pnl': profit_or_loss,
                'capital_after_trade': current_capital,
                'reason_for_exit': "End of Backtest",
                'pattern_score_at_entry': open_position['pattern_score_at_entry']
            })
            logger.info(f"Closed open position at end of backtest for {symbol} at {last_price:.2f}. P&L: {profit_or_loss:.2f}. Capital: {current_capital:.2f}")
            open_position = None

        if not backtest_df_with_indicators.empty:
             portfolio_history.append({'timestamp': backtest_df_with_indicators.index[-1], 'capital': current_capital})
        else:
            portfolio_history.append({'timestamp': datetime.now(timezone.utc), 'capital': current_capital})

        final_capital = current_capital
        total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
        num_trades = len(trades)
        num_wins = len([t for t in trades if t['pnl'] > 0])
        win_rate = (num_wins / num_trades) * 100 if num_trades > 0 else 0.0
        total_pnl = sum(t['pnl'] for t in trades)
        average_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0.0

        # Extended Financial Metrics
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        daily_returns_list = []

        if portfolio_history:
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
            portfolio_df.set_index('timestamp', inplace=True)
            portfolio_series = portfolio_df['capital'].sort_index()

            if not portfolio_series.empty:
                # Daily returns (or per-candle returns if data is not daily)
                # Resample to daily if enough data points, otherwise use per-change returns
                if len(portfolio_series) > 1:
                    # Attempt daily resampling if index spans multiple days
                    if (portfolio_series.index.max() - portfolio_series.index.min()).days > 0:
                         daily_capital = portfolio_series.resample('D').last().ffill()
                         daily_returns = daily_capital.pct_change().dropna()
                    else: # If less than a day of data, or single day, use simple pct_change
                         daily_returns = portfolio_series.pct_change().dropna()
                    daily_returns_list = daily_returns.tolist() # For JSON serialization if needed

                    if not daily_returns.empty:
                        mean_daily_return = daily_returns.mean()
                        std_daily_return = daily_returns.std()
                        if std_daily_return != 0 and not np.isnan(std_daily_return):
                            # Assuming 252 trading days for annualization if returns are daily,
                            # or sqrt(num_periods_in_year) if returns are for other fixed periods.
                            # For simplicity, using 365 if data is daily-like, otherwise adjust.
                            # If timeframe is less than daily, this annualization might be aggressive.
                            # Let's assume daily-like returns for now for Sharpe.
                            annualization_factor = 365 if (timeframe not in ['1m','5m','15m','30m','1h','2h','4h','6h','8h','12h']) else (365 * 24 * (3600 / TIMEFRAME_TO_SECONDS.get(timeframe, 3600)))
                            sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(annualization_factor)
                            if np.isnan(sharpe_ratio): sharpe_ratio = 0.0 # Handle potential NaN if std_daily_return was near zero then became NaN
                        else:
                            sharpe_ratio = 0.0 # Or np.inf if mean_daily_return is positive

                        # Max Drawdown
                        cumulative_returns = (1 + daily_returns).cumprod() # Based on daily returns
                        running_max = cumulative_returns.cummax()
                        drawdown = (cumulative_returns - running_max) / running_max
                        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
                        if np.isnan(max_drawdown): max_drawdown = 0.0
                    else: # Not enough daily returns to calculate sharpe/drawdown meaningfully
                        sharpe_ratio = 0.0
                        max_drawdown = 0.0
                else: # Only one portfolio entry or empty after processing
                    sharpe_ratio = 0.0
                    max_drawdown = 0.0
            else: # Portfolio series is empty
                sharpe_ratio = 0.0
                max_drawdown = 0.0
        else: # No portfolio history
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf # Or a large number like 99999 if no losses

        winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
        avg_profit_win_trade = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0

        losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
        avg_loss_lose_trade = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0

        logger.info(f"Backtest finished for {symbol} ({timeframe}), Pattern: {pattern_type}, Arch: {architecture_key}")
        logger.info(f"Initial Capital: {initial_capital:.2f}, Final Capital: {final_capital:.2f}")
        logger.info(f"Total Return: {total_return_pct:.2f}%")
        logger.info(f"Number of Trades: {num_trades}, Wins: {num_wins}, Win Rate: {win_rate:.2f}%")
        logger.info(f"Total P&L from trades: {total_pnl:.2f}, Avg P&L/trade: {average_pnl_per_trade:.2f}")
        logger.info(f"Avg Profit/Win: {avg_profit_win_trade:.2f}, Avg Loss/Losing: {avg_loss_lose_trade:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f if profit_factor != np.inf else 'inf'}")
        logger.info(f"Sharpe Ratio (annualized): {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")


        backtest_run_results = {
            'run_timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern_type': pattern_type,
            'architecture_key': architecture_key,
            'sequence_length_used': sequence_length,
            'backtest_start_date': backtest_start_date_str,
            'metrics': { # Grouping metrics under a sub-dictionary
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return_pct': total_return_pct,
                'total_pnl_from_trades': total_pnl,
                'num_trades': num_trades,
                'num_wins': num_wins,
                'num_losses': num_trades - num_wins,
                'win_rate_pct': win_rate,
                'loss_rate_pct': 100.0 - win_rate if num_trades > 0 else 0.0,
                'average_pnl_per_trade': average_pnl_per_trade,
                'avg_profit_per_winning_trade': avg_profit_win_trade,
                'avg_loss_per_losing_trade': avg_loss_lose_trade,
                'profit_factor': profit_factor if profit_factor != np.inf else None, # Storing inf as None for JSON
                'sharpe_ratio_annualized': sharpe_ratio if not np.isnan(sharpe_ratio) else None,
                'max_drawdown_pct': max_drawdown * 100 if not np.isnan(max_drawdown) else None, # Store as percentage
                # 'daily_returns': daily_returns_list, # Optional: if you want to store all daily returns
            },
            'params_used': { # Renamed from 'params' to 'params_used' for clarity
                'entry_threshold': entry_threshold,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'hold_duration_candles': hold_duration_candles,
                'stake_pct_capital': stake_pct_capital,
            },
            'trades': trades,
            'portfolio_history': portfolio_history
        }

        self._save_backtest_results(backtest_run_results)
        return backtest_run_results

    def _save_backtest_results(self, result_entry: dict):
        results = []
        if BACKTEST_RESULTS_FILE.exists():
            try:
                with open(BACKTEST_RESULTS_FILE, 'r') as f:
                    content = f.read()
                    if not content:
                        results = []
                    else:
                        data = json.loads(content)
                        if isinstance(data, list):
                            results = data
                        else:
                            results = [data]
            except json.JSONDecodeError:
                logger.error(f"Could not decode existing backtest results file {BACKTEST_RESULTS_FILE}. Starting fresh.")
                results = []
            except Exception as e:
                logger.error(f"Error reading {BACKTEST_RESULTS_FILE}: {e}. Starting fresh.", exc_info=True)
                results = []

        for trade in result_entry.get('trades', []):
            if isinstance(trade.get('entry_time'), pd.Timestamp):
                trade['entry_time'] = trade['entry_time'].isoformat()
            if isinstance(trade.get('exit_time'), pd.Timestamp):
                trade['exit_time'] = trade['exit_time'].isoformat()

        for item in result_entry.get('portfolio_history', []):
            if isinstance(item.get('timestamp'), pd.Timestamp):
                item['timestamp'] = item['timestamp'].isoformat()

        results.append(result_entry)

        try:
            with open(BACKTEST_RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            logger.info(f"Backtest results saved to {BACKTEST_RESULTS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}", exc_info=True)

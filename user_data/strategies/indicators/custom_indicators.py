# user_data/strategies/indicators/custom_indicators.py
import pandas as pd
from typing import Tuple

def calculate_rsi_pandas(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Pandas."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Replace inf values with 100 (where avg_loss is 0)
    rsi = rsi.replace([float('inf'), -float('inf')], 100.0)
    # Fill initial NaNs with 50 (neutral)
    rsi = rsi.fillna(50.0)
    return rsi

def calculate_macd_pandas(series: pd.Series, fast_p: int = 12, slow_p: int = 26, signal_p: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram using Pandas."""
    ema_fast = series.ewm(span=fast_p, adjust=False, min_periods=fast_p).mean()
    ema_slow = series.ewm(span=slow_p, adjust=False, min_periods=slow_p).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_p, adjust=False, min_periods=signal_p).mean()
    macd_hist = macd_line - signal_line

    return macd_line, signal_line, macd_hist

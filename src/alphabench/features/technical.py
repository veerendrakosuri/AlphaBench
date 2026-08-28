from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    # Normalise by price: a raw MACD of 2.0 means something different at $10 vs $500.
    return pd.DataFrame(
        {
            "macd": line / close,
            "macd_signal": sig / close,
            "macd_hist": (line - sig) / close,
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / close


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper, lower = ma + k * sd, ma - k * sd
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_pctb": (close - lower) / width,
            "bb_width": (upper - lower) / ma,
        }
    )


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3
) -> pd.DataFrame:
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    rng = (hh - ll).replace(0.0, np.nan)
    k = 100.0 * (close - ll) / rng
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(smooth).mean()})


def obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    return obv.diff(window) / volume.rolling(window).sum().replace(0.0, np.nan)

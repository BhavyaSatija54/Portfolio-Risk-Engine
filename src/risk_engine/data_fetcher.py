"""
Data Fetcher Module — Production Grade

Institutional-quality market data retrieval with defensive validation,
missing-data imputation, and strict type contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_OBSERVATIONS: int = 30  # Minimum observations required for any analysis
_MAX_MISSING_PCT: float = 0.10  # Warn if >10% missing per ticker
_CALENDAR_DAYS_BUFFER: float = 1.45  # Multiplier for trading->calendar conversion


@dataclass(frozen=True)
class DataStats:
    """Immutable summary statistics for fetched data."""

    ticker: str
    n_obs: int
    n_missing: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    annualized_return: float
    annualized_vol: float
    min_return: float
    max_return: float
    first_date: pd.Timestamp
    last_date: pd.Timestamp


class DataFetcher:
    """
    Retrieves and preprocesses adjusted-close price histories from Yahoo Finance.

    Guarantees after ``fetch_data`` succeeds:
    * No NaNs in price data (forward-filled up to 5 days, then dropped)
    * All requested tickers present with at least ``_MIN_OBSERVATIONS`` rows
    * Returns aligned to business days
    """

    def __init__(
        self,
        tickers: List[str],
        lookback_days: int = 7560,
        end_date: Optional[str] = None,
    ) -> None:
        self.tickers: List[str] = sorted({t.upper().strip() for t in tickers})
        if not self.tickers:
            raise ValueError("Tickers list must not be empty")

        self.lookback_days: int = int(lookback_days)
        if self.lookback_days < _MIN_OBSERVATIONS:
            raise ValueError(
                f"lookback_days must be >= {_MIN_OBSERVATIONS}, got {lookback_days}"
            )

        self.end_date: pd.Timestamp = (
            pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()
        )
        self.start_date: pd.Timestamp = self._calc_start()

        # Populated by fetch_data
        self.price_data: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_data(self) -> pd.DataFrame:
        """Fetch adjusted close prices; validate and compute returns."""
        logger.info(
            "Fetching %d tickers: %s  (%s → %s)",
            len(self.tickers),
            self.tickers,
            self.start_date.date(),
            self.end_date.date(),
        )

        raw = self._download()
        clean = self._sanitize(raw)
        clean = self._impute(clean)
        self._validate(clean)

        self.price_data = clean
        self.returns_data = self._compute_returns(clean)

        logger.info(
            "Data ready: %d obs × %d assets  (%s → %s)",
            len(clean),
            len(clean.columns),
            clean.index[0].date(),
            clean.index[-1].date(),
        )
        return self.price_data

    def get_prices(self) -> pd.DataFrame:
        if self.price_data is None:
            raise RuntimeError("fetch_data() must be called first")
        return self.price_data

    def get_returns(self) -> pd.DataFrame:
        if self.returns_data is None:
            raise RuntimeError("fetch_data() must be called first")
        return self.returns_data

    def get_stats(self) -> List[DataStats]:
        """Per-ticker descriptive statistics."""
        returns = self.get_returns()
        stats: List[DataStats] = []
        for col in returns.columns:
            r = returns[col]
            stats.append(
                DataStats(
                    ticker=col,
                    n_obs=len(r),
                    n_missing=r.isna().sum(),
                    mean=float(r.mean()),
                    std=float(r.std()),
                    skewness=float(r.skew()),
                    kurtosis=float(r.kurtosis()),
                    annualized_return=float(r.mean() * 252),
                    annualized_vol=float(r.std() * np.sqrt(252)),
                    min_return=float(r.min()),
                    max_return=float(r.max()),
                    first_date=r.index[0],
                    last_date=r.index[-1],
                )
            )
        return stats

    def get_covariance(self, annualized: bool = True) -> pd.DataFrame:
        returns = self.get_returns()
        cov = returns.cov()
        if annualized:
            cov *= 252
        return cov

    def get_correlation(self) -> pd.DataFrame:
        return self.get_returns().corr()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _calc_start(self) -> pd.Timestamp:
        """Estimate start calendar date with holiday/weekend buffer."""
        cal_days = int(self.lookback_days * _CALENDAR_DAYS_BUFFER)
        return self.end_date - timedelta(days=cal_days)

    def _download(self) -> pd.DataFrame:
        """Download from Yahoo Finance.  Handles both single- and multi-ticker."""
        try:
            data = yf.download(
                tickers=self.tickers if len(self.tickers) > 1 else self.tickers[0],
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise ConnectionError(f"Yahoo Finance download failed: {exc}") from exc

        if data.empty:
            raise ValueError("Yahoo Finance returned empty dataset")

        # Normalise column structure
        if len(self.tickers) == 1:
            prices = pd.DataFrame(
                {self.tickers[0]: data["Close"].values},
                index=data.index,
            )
        else:
            if "Close" not in data.columns.levels[0]:
                raise ValueError(f"Unexpected yfinance columns: {data.columns}")
            prices = data["Close"].copy()

        prices.index = pd.to_datetime(prices.index)
        return prices

    @staticmethod
    def _sanitize(prices: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that are all-NaN and trim to requested lookback."""
        prices = prices.dropna(how="all")
        if len(prices) == 0:
            raise ValueError("All price data is NaN after initial drop")

        # Trim to exact requested lookback from the end
        max_lookback = 7560  # ~30 years
        if len(prices) > max_lookback:
            prices = prices.iloc[-max_lookback:]

        # Ensure numeric
        prices = prices.apply(pd.to_numeric, errors="coerce")
        return prices

    @staticmethod
    def _impute(prices: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill up to 5 days; any remaining NaNs are dropped."""
        prices = prices.ffill(limit=5)
        # Drop rows with any remaining NaNs
        before = len(prices)
        prices = prices.dropna(how="any")
        dropped = before - len(prices)
        if dropped:
            logger.warning("Dropped %d rows after imputation due to remaining NaNs", dropped)
        return prices

    def _validate(self, prices: pd.DataFrame) -> None:
        """Run strict data-quality checks."""
        if len(prices) < _MIN_OBSERVATIONS:
            raise ValueError(
                f"Only {len(prices)} observations available; "
                f"minimum is {_MIN_OBSERVATIONS}"
            )

        missing_tickers = set(self.tickers) - set(prices.columns)
        if missing_tickers:
            raise ValueError(f"No data returned for: {sorted(missing_tickers)}")

        for col in prices.columns:
            pct_missing = prices[col].isna().mean()
            if pct_missing > _MAX_MISSING_PCT:
                logger.warning(
                    "%s: %.1f%% missing even after imputation", col, pct_missing * 100
                )

        # Detect stale prices (unchanged for >20 consecutive days)
        for col in prices.columns:
            unchanged = prices[col].diff().eq(0).astype(int).groupby(
                (prices[col].diff().ne(0)).cumsum()
            ).transform("sum")
            max_stale = unchanged.max()
            if max_stale > 20:
                logger.warning(
                    "%s: detected %d consecutive unchanged prices — possible stale data",
                    col,
                    max_stale,
                )

    @staticmethod
    def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Compute log-returns; drop the first NaN row."""
        returns = np.log(prices / prices.shift(1)).dropna()
        # Sanity check: no infinities
        if np.isinf(returns.values).any():
            n_inf = np.isinf(returns.values).sum()
            logger.warning("Replacing %d infinite returns with NaN then dropping", n_inf)
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns
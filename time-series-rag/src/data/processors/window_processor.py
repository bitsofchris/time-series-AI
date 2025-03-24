"""
Window processor for time series data.
Handles window slicing and normalization of price data.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import text
from database.connection import get_db_engine


class WindowProcessor:
    """Processes time series data into overlapping windows and normalizes them."""

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 10,
        db_url: Optional[str] = None,
    ):
        """
        Initialize the window processor.

        Args:
            window_size: Number of bars in each window
            stride: Number of bars to move forward for next window
            db_url: Database URL for TimescaleDB connection
        """
        self.window_size = window_size
        self.stride = stride
        self.db_url = db_url
        self.engine = get_db_engine(db_url)

    def fetch_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data from TimescaleDB for given symbols and timeframe.

        Args:
            symbols: List of ticker symbols
            timeframe: Time interval (e.g., '1day', '1hour')
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            Dictionary mapping symbols to their price DataFrames
        """
        # Print database connection info for debugging
        print(f"Database URL: {self.db_url}")
        print(f"Fetching data for symbols: {symbols}, timeframe: {timeframe}")
        print(f"Date range: {start_date or 'earliest'} to {end_date or 'latest'}")

        query = """
            SELECT time, symbol, open, high, low, close, volume
            FROM raw_price_data
            WHERE symbol = ANY(:symbols)
            AND timeframe = :timeframe
        """
        params = {"symbols": symbols, "timeframe": timeframe}

        if start_date:
            query += " AND time >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND time <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY time"
        print("Query:", query)
        print("Params:", params)

        try:
            with self.engine.connect() as conn:
                # Test connection with simple query
                test_result = conn.execute(text("SELECT 1 as test")).fetchone()
                print(f"Connection test result: {test_result}")

                # Execute the actual query
                df = pd.read_sql(text(query), conn, params=params)
                print(f"Query returned {len(df)} rows")

                if df.empty:
                    print("WARNING: Query returned no data!")
                else:
                    print(f"Data sample: \n{df.head(2)}")

                # Split into separate DataFrames by symbol
                result = {symbol: df[df["symbol"] == symbol] for symbol in symbols}
                return result
        except Exception as e:
            print(f"Error executing query: {e}")
            # Return empty dict if query fails
            return {symbol: pd.DataFrame() for symbol in symbols}

    def create_windows(
        self, df: pd.DataFrame, price_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Create overlapping windows from time series data.

        Args:
            df: DataFrame with price data
            price_columns: List of price columns to include in windows
                         (default: ['open', 'high', 'low', 'close'])

        Returns:
            Tuple of (windowed_data, window_start_times)
        """
        if price_columns is None:
            price_columns = ["open", "high", "low", "close"]

        # Ensure data is sorted by time
        df = df.sort_values("time")

        # Extract price data
        price_data = df[price_columns].values

        # Calculate number of windows
        n_windows = (len(df) - self.window_size) // self.stride + 1

        # Initialize arrays for windows and timestamps
        windows = np.zeros((n_windows, self.window_size, len(price_columns)))
        window_starts = []

        # Create windows
        for i in range(n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            windows[i] = price_data[start_idx:end_idx]
            window_starts.append(df["time"].iloc[start_idx])

        return windows, window_starts

    def normalize_windows(
        self,
        windows: np.ndarray,
        method: str = "zscore",
        include_volume: bool = False,
    ) -> np.ndarray:
        """
        Normalize price windows using specified method.

        Args:
            windows: Array of price windows
            method: Normalization method ('zscore' or 'minmax')
            include_volume: Whether to include volume in normalization

        Returns:
            Normalized windows array
        """
        if method not in ["zscore", "minmax"]:
            raise ValueError("Normalization method must be 'zscore' or 'minmax'")

        # Reshape to 2D for normalization
        n_windows, n_bars, n_features = windows.shape
        windows_2d = windows.reshape(-1, n_features)

        if method == "zscore":
            # Calculate mean and std for each feature
            mean = np.mean(windows_2d, axis=0)
            std = np.std(windows_2d, axis=0)
            # Avoid division by zero
            std[std == 0] = 1
            normalized = (windows_2d - mean) / std
        else:  # minmax
            min_vals = np.min(windows_2d, axis=0)
            max_vals = np.max(windows_2d, axis=0)
            # Avoid division by zero
            max_vals[max_vals == min_vals] = 1
            normalized = (windows_2d - min_vals) / (max_vals - min_vals)

        # Reshape back to 3D
        return normalized.reshape(n_windows, n_bars, n_features)

    def process_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[datetime], Dict]:
        """
        Process a single symbol's data into normalized windows.

        Args:
            symbol: Ticker symbol
            timeframe: Time interval
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            Tuple of (normalized_windows, window_start_times, metadata)
        """
        # Fetch data
        data_dict = self.fetch_data(
            symbols=[symbol],
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        df = data_dict[symbol]

        if df.empty:
            raise ValueError(f"No data found for {symbol}")

        # Create windows
        windows, window_starts = self.create_windows(df)

        # Normalize windows
        normalized_windows = self.normalize_windows(windows)

        # Create metadata
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "window_size": self.window_size,
            "stride": self.stride,
            "n_windows": len(normalized_windows),
            "start_date": df["time"].min(),
            "end_date": df["time"].max(),
        }

        return normalized_windows, window_starts, metadata

    def process_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Tuple[np.ndarray, List[datetime], Dict]]:
        """
        Process multiple symbols' data into normalized windows.

        Args:
            symbols: List of ticker symbols
            timeframe: Time interval
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            Dictionary mapping symbols to their processed data
        """
        results = {}
        for symbol in symbols:
            try:
                windows, starts, metadata = self.process_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[symbol] = (windows, starts, metadata)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        return results

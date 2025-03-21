"""
YFinance data fetcher for time-series-rag project.
Fetches market data from Yahoo Finance and inserts it into TimescaleDB.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import yfinance as yf

# Import pandas to use with DataFrames
import pandas as pd
from database.connection import get_db_engine

# Timeframe mapping - yfinance interval to our standard format
TIMEFRAME_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1hour",
    "1h": "1hour",
    "1d": "1day",
    "1wk": "1week",
}


class YFinanceFetcher:
    """Fetches market data from Yahoo Finance and inserts it into TimescaleDB."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the fetcher with database connection.

        Args:
            db_url: SQLAlchemy database URL. If None, uses DATABASE_URL env variable.
        """
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.engine = get_db_engine(self.db_url)

    def fetch_and_store(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Fetch data for multiple symbols and store in TimescaleDB.

        Args:
            symbols: List of ticker symbols to fetch
            timeframe: Time interval ('1m','5m','15m','30m','60m','1h','1d','1wk')
            start_date: Start date for data fetch (default: 1 year ago)
            end_date: End date for data fetch (default: today)

        Returns:
            Dictionary with symbols as keys and count of rows inserted as values
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # Validate timeframe
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. Must be one of {TIMEFRAME_MAP.keys()}"
            )

        standard_timeframe = TIMEFRAME_MAP[timeframe]

        results = {}
        for symbol in symbols:
            print(f"Fetching {symbol} at {timeframe} timeframe...")

            # Fetch data from Yahoo Finance
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=timeframe,
                progress=False,
                group_by="ticker",  # Group by ticker to handle multiple symbols better
            )

            if data.empty:
                print(f"No data found for {symbol}")
                results[symbol] = 0
                continue

            # Prepare data for insertion
            df = data.copy()

            # Handle different column structures based on result
            if isinstance(df.columns, pd.MultiIndex):
                # Case 1: MultiIndex columns (e.g., when group_by='ticker')
                # Extract just the relevant price data, dropping the ticker level
                if len(symbols) == 1:
                    # For single symbol, we may have (Price, Symbol) structure
                    if symbol in df.columns.levels[-1]:
                        # Extract just this symbol's data and flatten columns
                        df = df.xs(symbol, axis=1, level=-1)
                    else:
                        # Just flatten the MultiIndex columns
                        df.columns = [col[0].lower() for col in df.columns]
                else:
                    # For multiple symbols in one call, select just this symbol
                    # This should not happen in our loop but adding for robustness
                    if symbol in df.columns.levels[0]:
                        df = df[symbol]

            # Ensure index has the right name
            if df.index.name is None or df.index.name != "time":
                df.index.name = "time"

            # Reset index to make 'time' a regular column
            df = df.reset_index()

            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Add symbol and timeframe columns
            df["symbol"] = symbol
            df["timeframe"] = standard_timeframe

            # Select and order columns to match our schema
            try:
                # Standard OHLCV column names
                raw_df = df[
                    [
                        "time",
                        "symbol",
                        "timeframe",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ]

                # Insert into raw_price_data table
                with self.engine.begin() as conn:
                    row_count = raw_df.to_sql(
                        "raw_price_data",
                        conn,
                        if_exists="append",
                        index=False,
                        method="multi",
                    )
                    results[symbol] = row_count
                    print(f"Inserted {row_count} rows for {symbol}")

            except KeyError as e:
                print(f"Error with columns for {symbol}: {e}")
                print(f"Available columns: {df.columns}")
                results[symbol] = 0

        return results


def main():
    """Example usage."""
    # Initialize the fetcher
    fetcher = YFinanceFetcher()

    # Define symbols and parameters
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    timeframe = "1d"

    # Fetch one year of daily data for these symbols
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    # Fetch and store the data
    results = fetcher.fetch_and_store(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"Total rows inserted: {sum(results.values())}")


if __name__ == "__main__":
    main()

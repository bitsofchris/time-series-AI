"""
YFinance data fetcher for time-series-rag project.
Fetches market data from Yahoo Finance and inserts it into TimescaleDB.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import yfinance as yf
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
            timeframe: Time interval ('1m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk')
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
                f"Invalid timeframe: {timeframe}. Must be one of {list(TIMEFRAME_MAP.keys())}"
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
            )

            if data.empty:
                print(f"No data found for {symbol}")
                results[symbol] = 0
                continue

            # Prepare data for insertion
            df = data.copy()
            df.index.name = "time"
            df = df.reset_index()

            # Rename columns to match our schema
            new_columns = []
            for col in df.columns:
                if isinstance(col, tuple):
                    # Join tuple elements with underscore if it's a tuple
                    new_col = "_".join([str(part).lower() for part in col if part])
                else:
                    new_col = str(col).lower()
                new_columns.append(new_col)

            df.columns = new_columns

            # Add symbol and timeframe columns
            df["symbol"] = symbol
            df["timeframe"] = standard_timeframe

            # Select and order columns to match our schema
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

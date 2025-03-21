#!/usr/bin/env python
"""
Script to fetch market data from Yahoo Finance and store in TimescaleDB.
Use this script to populate the database with historical price data.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetchers.yfinance_fetcher import YFinanceFetcher
from database.connection import get_connection_string


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch market data from Yahoo Finance and store in TimescaleDB"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )

    parser.add_argument(
        "--timeframe",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk"],
        help="Timeframe for data (default: 1d)",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data to fetch (default: 365)",
    )

    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization step"
    )

    parser.add_argument(
        "--db-url",
        type=str,
        help="Database URL (default: uses DATABASE_URL env variable)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Parse symbols from comma-separated string
    symbols = [s.strip() for s in args.symbols.split(",")]

    # Get database URL
    db_url = args.db_url or os.environ.get("DATABASE_URL") or get_connection_string()

    # Initialize the fetcher
    fetcher = YFinanceFetcher(db_url=db_url)

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print(
        f"Fetching {args.timeframe} data for {len(symbols)} symbols: {', '.join(symbols)}"
    )
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    # Fetch and store the data
    results = fetcher.fetch_and_store(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    # Print results
    for symbol, count in results.items():
        print(f"{symbol}: {count} rows inserted")

    print(f"Total rows inserted: {sum(results.values())}")


if __name__ == "__main__":
    main()

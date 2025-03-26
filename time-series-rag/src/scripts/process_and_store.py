"""
Script to process market data and store embeddings in LanceDB.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import sqlalchemy as sa
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors.embedding_store import EmbeddingStore
from database.connection import get_connection_string


def get_all_symbols(db_url=None):
    """Get all available symbols from the database."""
    # Create a connection to the database
    db_url = db_url or get_connection_string()

    # Create SQLAlchemy engine
    engine = sa.create_engine(db_url)

    try:
        # Query for distinct symbols from raw_price_data table
        query = "SELECT DISTINCT symbol FROM raw_price_data"
        df = pd.read_sql(query, engine)

        if df.empty:
            return []

        # Return sorted list of symbols
        return sorted(df["symbol"].tolist())
    except Exception as e:
        print(f"Error fetching symbols: {str(e)}")
        return []


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process market data and store embeddings"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of stock symbols to process (default: all available symbols)",
    )
    parser.add_argument(
        "--timeframe", default="1day", help="Time interval (e.g., '1day', '1hour')"
    )
    parser.add_argument("--start-date", help="Start date in ISO format (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date in ISO format (YYYY-MM-DD)")
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of history to process (default: 5)",
    )
    parser.add_argument(
        "--window-size", type=int, default=20, help="Number of bars in each window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Number of bars to move forward for next window",
    )
    parser.add_argument(
        "--db-url",
        help="Database URL for TimescaleDB connection",
        default="postgresql://timeseriesrag:devpassword@localhost:5432/market_data",
    )
    parser.add_argument(
        "--lancedb-uri", help="URI for LanceDB storage", default="./lance-data/lancedb"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    return parser.parse_args()


def main():
    """Main function to process market data and store embeddings."""
    args = parse_args()

    # Get database URL if not provided
    if not args.db_url:
        args.db_url = get_connection_string()
        print(f"Using database URL from environment: {args.db_url}")
    else:
        print(f"Using provided database URL: {args.db_url}")

    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Default to specified years of data
        start = datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(
            days=args.years * 365
        )
        args.start_date = start.strftime("%Y-%m-%d")

    # Get all symbols if not provided
    if not args.symbols:
        print("No symbols provided. Fetching all available symbols...")
        args.symbols = get_all_symbols(args.db_url)
        if not args.symbols:
            print("Error: No symbols found in the database.")
            sys.exit(1)

        # Show first few symbols
        first_symbols = ", ".join(args.symbols[:5])
        suffix = "..." if len(args.symbols) > 5 else ""
        print(f"Found {len(args.symbols)} symbols: {first_symbols}{suffix}")

        # Confirm with user if large number of symbols
        if len(args.symbols) > 10 and not args.verbose:
            confirm = input(
                f"You are about to process {len(args.symbols)} symbols. Continue? (y/n): "
            )
            if confirm.lower() != "y":
                print("Cancelled by user.")
                sys.exit(0)

    # Initialize the embedding store
    store = EmbeddingStore(
        window_size=args.window_size,
        stride=args.stride,
        db_url=args.db_url,
        uri=args.lancedb_uri,
    )

    # Process and store the data
    print(
        f"Processing {len(args.symbols)} symbols from "
        f"{args.start_date} to {args.end_date}"
    )
    stored_counts = store.process_and_store(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Print results
    print("\nProcessing complete!")
    print("Windows stored per symbol:")
    total_windows = 0
    for symbol, count in stored_counts.items():
        total_windows += count
        if args.verbose or len(stored_counts) <= 10:
            print(f"{symbol}: {count} windows")

    if len(stored_counts) > 10 and not args.verbose:
        print(
            f"Processed {len(stored_counts)} symbols with {total_windows} total windows"
        )
        print("Use --verbose to see details for each symbol")


if __name__ == "__main__":
    main()

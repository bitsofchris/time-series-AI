"""
Script to process market data and store embeddings in LanceDB.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors.embedding_store import EmbeddingStore
from database.connection import get_connection_string


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process market data and store embeddings"
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True, help="List of stock symbols to process"
    )
    parser.add_argument(
        "--timeframe", default="1day", help="Time interval (e.g., '1day', '1hour')"
    )
    parser.add_argument("--start-date", help="Start date in ISO format (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date in ISO format (YYYY-MM-DD)")
    parser.add_argument(
        "--window-size", type=int, default=20, help="Number of bars in each window"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Number of bars to move forward for next window",
    )
    parser.add_argument("--db-url", help="Database URL for TimescaleDB connection")
    parser.add_argument("--lancedb-uri", help="URI for LanceDB storage")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    return parser.parse_args()


def main():
    """Main function to process market data and store embeddings."""
    args = parse_args()

    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Default to 1 year of data
        start = datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=365)
        args.start_date = start.strftime("%Y-%m-%d")

    # Get database URL if not provided
    if not args.db_url:
        args.db_url = get_connection_string()
        print(f"Using database URL from environment: {args.db_url}")
    else:
        print(f"Using provided database URL: {args.db_url}")

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
    for symbol, count in stored_counts.items():
        print(f"{symbol}: {count} windows")


if __name__ == "__main__":
    main()

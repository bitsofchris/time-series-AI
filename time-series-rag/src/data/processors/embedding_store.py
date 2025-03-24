"""
Module for storing normalized time series windows in LanceDB.
"""

import os
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import lancedb
import pyarrow as pa
from datetime import datetime

from data.processors.window_processor import WindowProcessor


class EmbeddingStore:
    """Handles storage of normalized time series windows in LanceDB."""

    def __init__(
        self,
        table_name: str = "time_series_windows",
        uri: Optional[str] = None,
        window_size: int = 20,
        stride: int = 10,
        db_url: Optional[str] = None,
    ):
        """
        Initialize the embedding store.

        Args:
            table_name: Name of the LanceDB table
            uri: URI for LanceDB storage (default: ./data/lancedb)
            window_size: Number of bars in each window
            stride: Number of bars to move forward for next window
            db_url: Database URL for TimescaleDB connection
        """
        self.table_name = table_name
        self.uri = uri or os.path.join(os.getcwd(), "data", "lancedb")
        self.window_size = window_size
        self.window_processor = WindowProcessor(
            window_size=window_size, stride=stride, db_url=db_url
        )
        self.db = self._connect_db()
        self._ensure_table_exists()

    def _connect_db(self):
        """Connect to the LanceDB database."""
        # Create directory if it doesn't exist
        if not os.path.exists(self.uri):
            os.makedirs(self.uri, exist_ok=True)

        # Connect to the database
        return lancedb.connect(self.uri)

    def _ensure_table_exists(self):
        """Create the LanceDB table if it doesn't exist."""
        # Check if table exists
        if self.table_name in self.db:
            print(f"Table '{self.table_name}' already exists.")
            self.table = self.db.open_table(self.table_name)
        else:
            print(f"Table '{self.table_name}' does not exist. Creating it now.")

            # Convert 2D array to 1D for sample data
            # Flatten the 20x4 array to a single vector of length 80
            vector_size = self.window_size * 4  # window_size * features_per_window
            sample_vector = np.zeros(vector_size, dtype=np.float32)

            # Create a sample dataframe with the correct schema
            sample_data = pd.DataFrame(
                {
                    "id": [0],
                    "symbol": ["SAMPLE"],
                    "timeframe": ["1day"],
                    "window_start": [pd.Timestamp.now()],
                    "window_end": [pd.Timestamp.now()],
                    "vector": [sample_vector],  # Use a flattened vector
                    "metadata": [json.dumps({"sample": True})],
                }
            )

            # Define schema with vector type
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("symbol", pa.string()),
                    pa.field("timeframe", pa.string()),
                    pa.field("window_start", pa.timestamp("ns")),
                    pa.field("window_end", pa.timestamp("ns")),
                    pa.field(
                        "vector", pa.list_(pa.float32(), vector_size)
                    ),  # Fixed-length vector
                    pa.field("metadata", pa.string()),
                ]
            )

            # Create the table with sample data and schema
            self.table = self.db.create_table(
                self.table_name, data=sample_data, schema=schema, mode="overwrite"
            )

    def process_and_store(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Process symbols and store their normalized windows in LanceDB.

        Args:
            symbols: List of ticker symbols
            timeframe: Time interval (e.g., '1day', '1hour')
            start_date: Start date in ISO format
            end_date: End date in ISO format

        Returns:
            Dictionary mapping symbols to number of windows stored
        """
        # Process the symbols
        results = self.window_processor.process_multiple_symbols(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        # Store the results
        stored_counts = {}
        for symbol, (windows, starts, metadata) in results.items():
            count = self._store_windows(symbol, windows, starts, metadata)
            stored_counts[symbol] = count

        return stored_counts

    def _store_windows(
        self, symbol: str, windows: np.ndarray, starts: List[datetime], metadata: Dict
    ) -> int:
        """
        Store normalized windows for a symbol in LanceDB.

        Args:
            symbol: Ticker symbol
            windows: Array of normalized windows
            starts: List of window start times
            metadata: Additional metadata for the windows

        Returns:
            Number of windows stored
        """
        # Prepare data for storage
        records = []
        for i, (window, start) in enumerate(zip(windows, starts)):
            end = start + (metadata["window_size"] - 1) * pd.Timedelta(
                metadata["timeframe"]
            )
            # Flatten the 2D window to 1D vector
            flattened_vector = window.flatten().astype(np.float32)

            records.append(
                {
                    "id": i,
                    "symbol": symbol,
                    "timeframe": metadata["timeframe"],
                    "window_start": start,
                    "window_end": end,
                    "vector": flattened_vector,  # Store as flattened vector
                    "metadata": json.dumps(metadata),  # Convert metadata to JSON string
                }
            )

        # Convert to DataFrame for LanceDB
        df = pd.DataFrame(records)

        # Add to LanceDB table
        self.table.add(df)

        return len(records)

    def get_windows(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[datetime], List[Dict]]:
        """
        Retrieve windows for a symbol from LanceDB.

        Args:
            symbol: Ticker symbol
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of windows to return

        Returns:
            Tuple of (windows_array, start_times, metadata_list)
        """
        # Build query
        query = f"symbol = '{symbol}'"
        if start_date:
            query += f" AND window_start >= '{start_date}'"
        if end_date:
            query += f" AND window_end <= '{end_date}'"

        # Execute query
        results = self.table.search(query).limit(limit if limit else None).to_pandas()

        if results.empty:
            return np.array([]), [], []

        # Convert results to expected format - reshape vectors back to 2D windows
        windows = np.array(
            [vector.reshape(self.window_size, 4) for vector in results["vector"].values]
        )
        starts = results["window_start"].tolist()
        metadata = [
            json.loads(m) for m in results["metadata"].values
        ]  # Parse JSON strings

        return windows, starts, metadata

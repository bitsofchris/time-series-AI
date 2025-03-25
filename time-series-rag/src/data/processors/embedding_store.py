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
from datetime import datetime, timedelta

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

            # Create a vector index for similarity search
            self.table.create_index(["vector"], index_type="HNSW", metric_type="L2")

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

            # Convert datetime objects in metadata to ISO format strings
            serializable_metadata = metadata.copy()
            if "start_date" in serializable_metadata:
                serializable_metadata["start_date"] = serializable_metadata[
                    "start_date"
                ].isoformat()
            if "end_date" in serializable_metadata:
                serializable_metadata["end_date"] = serializable_metadata[
                    "end_date"
                ].isoformat()

            records.append(
                {
                    "id": i,
                    "symbol": symbol,
                    "timeframe": metadata["timeframe"],
                    "window_start": start,
                    "window_end": end,
                    "vector": flattened_vector,  # Store as flattened vector
                    "metadata": json.dumps(
                        serializable_metadata
                    ),  # Convert metadata to JSON string
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

    def find_similar_windows(
        self, vector: np.ndarray, n: int = 10, exclude_symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Find windows similar to the given vector.

        Args:
            vector: The query vector to find similar windows to
            n: Number of similar windows to return
            exclude_symbol: Optional symbol to exclude from results

        Returns:
            DataFrame with similar windows and their metadata
        """
        # Ensure the vector is properly formatted (flattened)
        if vector.ndim > 1:
            vector = vector.flatten().astype(np.float32)

        # Build vector similarity query
        # Use vector search with nearest neighbors
        query = self.table.search(vector=vector.tolist(), vector_column="vector")

        # Exclude specific symbol if requested
        if exclude_symbol:
            query = query.where(f"symbol != '{exclude_symbol}'")

        # Execute and return results
        results = query.limit(n).to_pandas()

        if not results.empty and "_distance" in results.columns:
            # Rename _distance to distance for consistency
            results["distance"] = results["_distance"]

        return results

    def find_similar_to_recent(
        self, symbol: str, n: int = 10, days_back: int = 7, exclude_self: bool = True
    ) -> pd.DataFrame:
        """
        Find windows similar to the most recent window of a specific symbol.

        Args:
            symbol: Symbol to find the most recent window for
            n: Number of similar windows to return
            days_back: Number of days to look back for the recent window
            exclude_self: Whether to exclude windows from the same symbol

        Returns:
            DataFrame with similar windows and their metadata
        """
        # Calculate the date range for recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get the most recent window for the symbol
        windows, starts, metadata = self.get_windows(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            limit=1,
        )

        # Check if we found any windows
        if len(windows) == 0:
            print(f"No recent windows found for {symbol} in the last {days_back} days")
            return pd.DataFrame()

        # Use the most recent window as the query vector
        query_vector = windows[0]

        # Find similar windows
        exclude = symbol if exclude_self else None
        return self.find_similar_windows(
            vector=query_vector, n=n, exclude_symbol=exclude
        )

    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols in the database.

        Returns:
            List of unique symbol names
        """
        # Query for distinct symbols
        results = self.table.to_pandas()
        if results.empty:
            return []

        return sorted(results["symbol"].unique().tolist())

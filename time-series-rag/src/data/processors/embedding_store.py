"""
Module for storing normalized time series windows in LanceDB.
"""

import os
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import lancedb
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

            # Create the table with sample data
            self.table = self.db.create_table(
                self.table_name, data=sample_data, mode="overwrite"
            )

            # Create a basic vector index for similarity search
            try:
                self.table.create_fts_index(
                    ["symbol"]
                )  # Create text index on symbol for filtering
                print("Created text index on 'symbol' column")

                # Create vector index for similarity search
                vector_params = {
                    "num_partitions": 256,
                    "num_sub_vectors": 96,
                }
                self.table.create_vector_index("vector", vector_params=vector_params)
                print("Created vector index on 'vector' column")
            except Exception as e:
                print(f"Warning: Could not create index: {str(e)}")
                print("Search may be slower without indexes")

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
        try:
            # Build filter query starting with search
            # We need an empty search to apply filters
            query = self.table.search()

            # Apply filters
            query = query.where(f"symbol = '{symbol}'")

            if start_date:
                # Convert string date to proper timestamp format
                start_timestamp = pd.to_datetime(start_date)
                query = query.where(
                    f"window_start >= to_timestamp('{start_timestamp}')"
                )
            if end_date:
                # Convert string date to proper timestamp format
                end_timestamp = pd.to_datetime(end_date)
                query = query.where(f"window_end <= to_timestamp('{end_timestamp}')")

            # Apply limit if provided
            if limit:
                query = query.limit(limit)

            # Get results
            results = query.to_pandas()

            if results.empty:
                return np.array([]), [], []

            # Convert results to expected format - reshape vectors back to 2D windows
            windows = np.array(
                [
                    vector.reshape(self.window_size, 4)
                    for vector in results["vector"].values
                ]
            )
            starts = results["window_start"].tolist()
            metadata = [
                json.loads(m) for m in results["metadata"].values
            ]  # Parse JSON strings

            return windows, starts, metadata

        except Exception as e:
            print(f"Error in get_windows: {str(e)}")
            # Fallback to loading all data and filtering in Python
            try:
                print("Trying fallback method for get_windows...")
                all_data = self.table.to_pandas()

                # Apply filters
                filtered = all_data[all_data["symbol"] == symbol]
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    filtered = filtered[filtered["window_start"] >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    filtered = filtered[filtered["window_end"] <= end_dt]

                # Apply limit
                if limit:
                    filtered = filtered.head(limit)

                if filtered.empty:
                    return np.array([]), [], []

                # Process results
                windows = np.array(
                    [
                        vector.reshape(self.window_size, 4)
                        for vector in filtered["vector"].values
                    ]
                )
                starts = filtered["window_start"].tolist()
                metadata = [json.loads(m) for m in filtered["metadata"].values]

                return windows, starts, metadata

            except Exception as fallback_e:
                print(f"Fallback method also failed: {str(fallback_e)}")
                return np.array([]), [], []

    def find_similar_windows(
        self,
        vector: np.ndarray,
        n: int = 10,
        exclude_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Find the N most similar windows to a given vector.

        Args:
            vector: The query vector (normalized window)
            n: Number of similar windows to return
            exclude_symbol: Optional symbol to exclude from results

        Returns:
            DataFrame with the N most similar windows and their metadata
        """
        # Ensure vector is the right shape (flatten if 2D)
        if len(vector.shape) > 1:
            query_vector = vector.flatten().astype(np.float32)
        else:
            query_vector = vector.astype(np.float32)

        try:
            # Use search method instead of nearest_neighbors
            query = self.table.search(query_vector).limit(n)

            # Apply filter if needed
            if exclude_symbol:
                query = query.where(f"symbol != '{exclude_symbol}'")

            # Get results and limit to n
            query = query.limit(n)
            results = query.to_pandas()

            if results.empty:
                print("No similar windows found")
                return pd.DataFrame()

            # Reshape vectors back to 2D for easier analysis
            results["window_data"] = results["vector"].apply(
                lambda v: (
                    v.reshape(self.window_size, 4)
                    if len(v) == self.window_size * 4
                    else v
                )
            )

            # Parse metadata from JSON strings
            results["parsed_metadata"] = results["metadata"].apply(json.loads)

            return results

        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            print("Trying fallback approach...")

            # Fallback to manual similarity calculation if needed
            try:
                # Get all data
                all_data = self.table.to_pandas()

                if exclude_symbol:
                    all_data = all_data[all_data["symbol"] != exclude_symbol]

                if all_data.empty:
                    return pd.DataFrame()

                # Calculate distances manually
                def euclidean_distance(v):
                    return np.linalg.norm(v - query_vector)

                all_data["distance"] = all_data["vector"].apply(euclidean_distance)

                # Sort by distance and take top n
                results = all_data.sort_values("distance").head(n)

                # Reshape vectors back to 2D for easier analysis
                results["window_data"] = results["vector"].apply(
                    lambda v: (
                        v.reshape(self.window_size, 4)
                        if len(v) == self.window_size * 4
                        else v
                    )
                )

                # Parse metadata from JSON strings
                results["parsed_metadata"] = results["metadata"].apply(json.loads)

                return results

            except Exception as fallback_e:
                print(f"Fallback search also failed: {str(fallback_e)}")
                return pd.DataFrame()

    def find_similar_to_symbol_recent(
        self,
        symbol: str,
        n: int = 10,
        exclude_self: bool = True,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """
        Find windows similar to a symbol's most recent window.

        Args:
            symbol: The ticker symbol to find the recent window for
            n: Number of similar windows to return
            exclude_self: Whether to exclude the same symbol from results
                If False, will include the same symbol but exclude the current window
            days_back: How many days to look back for the recent window

        Returns:
            DataFrame with the N most similar windows
        """
        # Get recent data for the symbol
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        windows, starts, metadata = self.get_windows(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=1,  # Just get the most recent window
        )

        # If no recent window found, return empty result
        if len(windows) == 0:
            print(f"No recent windows found for {symbol}")
            return pd.DataFrame()

        # Get the most recent window
        recent_window = windows[-1]  # Last window should be most recent
        recent_start = starts[-1]

        # Calculate window end date based on window size and timeframe
        if metadata and len(metadata) > 0:
            timeframe = metadata[0].get("timeframe", "1day")
            window_size = metadata[0].get("window_size", self.window_size)
            recent_end = recent_start + (window_size - 1) * pd.Timedelta(timeframe)
        else:
            # Fallback if metadata not available
            recent_end = recent_start + pd.Timedelta(days=self.window_size)

        # Find similar windows
        if exclude_self:
            # Complete exclude the symbol
            return self.find_similar_windows(
                vector=recent_window, n=n, exclude_symbol=symbol
            )
        else:
            # Include the same symbol but exclude the current window
            # We'll do this by filtering results post-search

            # Get more results than needed since we'll filter some out
            results = self.find_similar_windows(
                vector=recent_window, n=n * 2, exclude_symbol=None
            )

            if results.empty:
                return pd.DataFrame()

            # Filter out the current window by checking for date overlap
            def is_not_overlapping(row):
                window_start = row["window_start"]

                # If this is from a different symbol, keep it
                if row["symbol"] != symbol:
                    return True

                # For the same symbol, filter out the recent window
                # Check if the window's start date is the same as the recent window's start
                if window_start == recent_start:
                    return False

                # Also check if window's dates overlap with recent window
                window_end = None
                if (
                    "parsed_metadata" in row
                    and "window_size" in row["parsed_metadata"]
                    and "timeframe" in row["parsed_metadata"]
                ):
                    window_size = row["parsed_metadata"]["window_size"]
                    timeframe = row["parsed_metadata"]["timeframe"]
                    window_end = window_start + (window_size - 1) * pd.Timedelta(
                        timeframe
                    )
                else:
                    # Fallback
                    window_end = window_start + pd.Timedelta(days=self.window_size)

                # Check for overlap
                return not ((window_start <= recent_end and window_end >= recent_start))

            # Apply filter and limit to required number
            filtered_results = results[results.apply(is_not_overlapping, axis=1)].head(
                n
            )
            return filtered_results

    def get_most_recent_window(
        self, symbol: str, days_back: int = 30
    ) -> Tuple[Optional[np.ndarray], Optional[datetime], Optional[Dict]]:
        """
        Get the most recent window for a symbol.

        Args:
            symbol: The ticker symbol
            days_back: How many days to look back

        Returns:
            Tuple of (window_array, start_time, metadata)
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        windows, starts, metadata = self.get_windows(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=1,  # Just get the most recent window
        )

        if len(windows) == 0:
            return None, None, None

        return windows[-1], starts[-1], metadata[-1]

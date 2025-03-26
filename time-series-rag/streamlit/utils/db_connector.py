"""
Database connection and query utilities.
"""

import pandas as pd
import streamlit as st
from database.connection import get_connection_string
from data.processors.embedding_store import EmbeddingStore


def get_embedding_store():
    """Get an instance of the EmbeddingStore class."""
    from database.connection import get_connection_string

    db_url = st.session_state.get("db_url", get_connection_string())
    lancedb_uri = st.session_state.get(
        "lancedb_uri", "/Users/chris/repos/time-series-AI/lance-data/lancedb"
    )
    window_size = st.session_state.get("window_size", 5)
    stride = st.session_state.get("stride", 2)

    return EmbeddingStore(
        db_url=db_url,
        uri=lancedb_uri,
        window_size=window_size,
        stride=stride,
    )


def get_available_symbols():
    """Get a list of available symbols from the database."""
    store = get_embedding_store()

    try:
        # Using LanceDB API to get distinct symbols
        results = store.table.search().select(["symbol"]).to_pandas()

        if results.empty:
            return []

        # Get unique values and sort
        symbols = sorted(results["symbol"].unique().tolist())
        return symbols
    except Exception as e:
        st.error(f"Error fetching symbols: {str(e)}")
        return []

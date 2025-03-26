"""
Streamlit app for visualizing time series similarity search results.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add src directory to Python path for importing project modules
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from data.processors.embedding_store import EmbeddingStore
from data.processors.window_processor import WindowProcessor
from database.connection import get_connection_string


def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Time Series Similarity Search",
        page_icon="📊",
        layout="wide",
    )
    st.title("Time Series Similarity Search")


def get_embedding_store():
    """Get an instance of the EmbeddingStore class."""
    db_url = st.session_state.get("db_url", get_connection_string())
    lancedb_uri = st.session_state.get(
        "lancedb_uri", "/Users/chris/repos/time-series-AI/lance-data/lancedb"
    )
    window_size = st.session_state.get("window_size", 20)
    stride = st.session_state.get("stride", 10)

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
        # First get all records with symbol column
        results = store.table.search().select(["symbol"]).to_pandas()

        if results.empty:
            return []

        # Get unique values and sort
        symbols = sorted(results["symbol"].unique().tolist())
        return symbols
    except Exception as e:
        st.error(f"Error fetching symbols: {str(e)}")
        return []


def plot_window(window, title=None, height=300):
    """
    Plot a normalized time series window.

    Args:
        window: 2D array of shape (window_size, 4) with normalized OHLC data
        title: Optional title for the plot
        height: Height of the plot in pixels
    """
    # Create figure
    fig = go.Figure()

    # Add the 4 lines (Open, High, Low, Close)
    feature_names = ["Open", "High", "Low", "Close"]
    colors = ["blue", "green", "red", "purple"]

    for i, (feature, color) in enumerate(zip(feature_names, colors)):
        fig.add_trace(
            go.Scatter(
                y=window[:, i],
                mode="lines",
                name=feature,
                line=dict(color=color, width=2),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time Step",
        yaxis_title="Normalized Value",
    )

    return fig


def plot_multiple_windows(results, max_plots=5):
    """
    Plot multiple normalized time series windows from similarity results.

    Args:
        results: DataFrame with similarity search results
        max_plots: Maximum number of windows to plot
    """
    # Limit the number of plots
    results = results.head(max_plots)

    # Create figure for each result
    for i, row in results.iterrows():
        window = row["window_data"]
        symbol = row["symbol"]
        distance = row.get("distance", 0)
        start_date = row["window_start"].strftime("%Y-%m-%d")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.plotly_chart(
                plot_window(
                    window,
                    title=f"{symbol} (Start: {start_date}, Distance: {distance:.4f})",
                ),
                use_container_width=True,
            )

        with col2:
            # Display metadata
            st.subheader("Metadata")
            metadata = row["parsed_metadata"]
            for key, value in metadata.items():
                st.text(f"{key}: {value}")


def app_mode_most_similar():
    """App mode for finding N most similar windows."""
    st.header("Find Most Similar Windows")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbols = get_available_symbols()
        default_idx = 0 if symbols else None
        symbol = st.selectbox("Select Symbol", symbols, index=default_idx)

    with col2:
        limit = st.slider("Number of Results", min_value=1, max_value=20, value=5)

    with col3:
        search_button = st.button("Find Similar Windows")

    if not symbols:
        st.warning("No symbols found in the database. Please process some data first.")
        return

    if search_button and symbol:
        store = get_embedding_store()

        # Get the most recent window for the selected symbol
        window, start, metadata = store.get_most_recent_window(symbol)

        if window is None:
            st.error(f"No windows found for {symbol}")
            return

        st.subheader(f"Query Window: {symbol}")
        st.plotly_chart(
            plot_window(
                window, title=f"Query: {symbol} (Start: {start.strftime('%Y-%m-%d')})"
            ),
            use_container_width=True,
        )

        # Find similar windows
        results = store.find_similar_windows(window, n=limit)

        if results.empty:
            st.warning("No similar windows found.")
            return

        st.subheader("Similar Windows")
        plot_multiple_windows(results)


def app_mode_similar_to_recent():
    """App mode for finding windows similar to a symbol's recent window."""
    st.header("Find Windows Similar to Recent Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbols = get_available_symbols()
        default_idx = 0 if symbols else None
        source_symbol = st.selectbox("Source Symbol", symbols, index=default_idx)

    with col2:
        limit = st.slider("Number of Results", min_value=1, max_value=20, value=5)

    with col3:
        exclude_self = st.checkbox("Exclude Source Symbol", value=True)

    with col4:
        search_button = st.button("Find Similar to Recent")

    if not symbols:
        st.warning("No symbols found in the database. Please process some data first.")
        return

    if search_button and source_symbol:
        store = get_embedding_store()

        # Find windows similar to the recent window of the source symbol
        results = store.find_similar_to_symbol_recent(
            symbol=source_symbol,
            n=limit,
            exclude_self=exclude_self,
        )

        if results.empty:
            st.warning(f"No similar windows found for {source_symbol}'s recent data.")
            return

        st.subheader(f"Windows Similar to Recent {source_symbol}")
        plot_multiple_windows(results)


def app_settings():
    """App settings section."""
    st.header("Settings")

    col1, col2 = st.columns(2)

    with col1:
        db_url = st.text_input(
            "Database URL",
            value=st.session_state.get("db_url", get_connection_string()),
        )

        lancedb_uri = st.text_input(
            "LanceDB URI",
            value=st.session_state.get(
                "lancedb_uri", "/Users/chris/repos/time-series-AI/lance-data/lancedb"
            ),
        )

    with col2:
        window_size = st.number_input(
            "Window Size",
            min_value=5,
            max_value=100,
            value=st.session_state.get("window_size", 20),
        )

        stride = st.number_input(
            "Stride",
            min_value=1,
            max_value=window_size,
            value=st.session_state.get("stride", 10),
        )

    # Save settings to session state
    if st.button("Save Settings"):
        st.session_state.db_url = db_url
        st.session_state.lancedb_uri = lancedb_uri
        st.session_state.window_size = window_size
        st.session_state.stride = stride
        st.success("Settings saved!")


def main():
    """Main function for the Streamlit app."""
    configure_page()

    # App mode selection
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Find Most Similar Windows", "Find Similar to Recent", "Settings"],
    )

    # Run the selected app mode
    if app_mode == "Find Most Similar Windows":
        app_mode_most_similar()
    elif app_mode == "Find Similar to Recent":
        app_mode_similar_to_recent()
    elif app_mode == "Settings":
        app_settings()

    # Display app info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app demonstrates time series similarity search using vector embeddings. "
        "Normalized price data is stored in LanceDB and can be queried for similarity."
    )


if __name__ == "__main__":
    main()

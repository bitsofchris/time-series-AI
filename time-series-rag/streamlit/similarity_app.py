"""
Streamlit app for visualizing time series similarity.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.processors.embedding_store import EmbeddingStore
from data.processors.window_processor import WindowProcessor
from database.connection import get_connection_string

# Page config
st.set_page_config(
    page_title="Time Series Similarity Explorer",
    page_icon="📈",
    layout="wide",
)


# Initialize connection to embedding store
@st.cache_resource
def get_embedding_store():
    db_url = get_connection_string()
    return EmbeddingStore(db_url=db_url)


# Plot a single window of time series data
def plot_window(window, title, start_time=None):
    # Create figure
    fig = go.Figure()

    # Assuming window has 4 features: open, high, low, close
    dates = [
        start_time + timedelta(days=i) if start_time else i for i in range(len(window))
    ]

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=window[:, 0],
            high=window[:, 1],
            low=window[:, 2],
            close=window[:, 3],
            name="Price",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Normalized Price",
        height=400,
    )

    return fig


# Function to denormalize window for better visualization if needed
def denormalize_window(window, metadata=None):
    # For z-score normalized data, this would be roughly in the -2 to +2 range
    # Let's scale it to make visualization easier - this is just for display
    # In a real app, you might want to denormalize to actual price values
    return window


# Main function
def main():
    st.title("Time Series Similarity Explorer")

    try:
        # Get embedding store
        store = get_embedding_store()

        # Get available symbols
        symbols = store.get_available_symbols()

        if not symbols:
            st.error(
                "No data found in the database. Please run the process_and_store.py script first."
            )
            return

        # Sidebar for mode selection
        st.sidebar.title("Mode Selection")
        mode = st.sidebar.radio(
            "Select mode:", ["Find Similar Patterns", "Compare to Recent Data"]
        )

        if mode == "Find Similar Patterns":
            # Mode 1: Find similar patterns across all data
            st.header("Find Similar Patterns")

            col1, col2 = st.columns(2)
            with col1:
                # Select a symbol
                symbol = st.selectbox("Select symbol:", symbols)

                # Select time range
                days_back = st.slider("Days to look back:", 7, 365, 30)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)

                # Format dates for display
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                st.write(f"Date range: {start_str} to {end_str}")

            with col2:
                # Number of windows to retrieve
                limit = st.slider("Number of windows to retrieve:", 1, 20, 5)

                # Number of similar windows to find
                n_similar = st.slider("Number of similar windows to find:", 1, 20, 5)

                # Exclude self or not
                exclude_self = st.checkbox(
                    "Exclude this symbol from results", value=True
                )

            # Button to find windows
            if st.button("Find Windows"):
                with st.spinner("Fetching windows..."):
                    windows, starts, metadata = store.get_windows(
                        symbol=symbol,
                        start_date=start_str,
                        end_date=end_str,
                        limit=limit,
                    )

                    if len(windows) == 0:
                        st.error(
                            f"No windows found for {symbol} in the selected date range."
                        )
                        return

                    # Display the windows
                    st.subheader(f"Selected Windows for {symbol}")

                    # Create a container for query windows
                    query_container = st.container()
                    with query_container:
                        # Display each window with a button to find similar windows
                        for i, (window, start) in enumerate(zip(windows, starts)):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                window_display = denormalize_window(window)
                                fig = plot_window(
                                    window_display,
                                    f"Window {i+1}: {start.strftime('%Y-%m-%d')}",
                                    start,
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                meta_str = "\n".join(
                                    [
                                        f"{k}: {v}"
                                        for k, v in metadata[i].items()
                                        if k not in ["sample"]
                                    ]
                                )
                                st.text_area(f"Metadata {i+1}", meta_str, height=300)

                                if st.button(
                                    f"Find Similar to Window {i+1}", key=f"btn_{i}"
                                ):
                                    with st.spinner(
                                        f"Finding windows similar to {i+1}..."
                                    ):
                                        similar_df = store.find_similar_windows(
                                            vector=window,
                                            n=n_similar,
                                            exclude_symbol=(
                                                symbol if exclude_self else None
                                            ),
                                        )

                                        if similar_df.empty:
                                            st.error("No similar windows found.")
                                            return

                                        # Display similar windows
                                        st.subheader(
                                            f"Windows Similar to {symbol} Window {i+1}"
                                        )

                                        # Create columns for similar windows
                                        similar_container = st.container()
                                        with similar_container:
                                            for j, row in similar_df.iterrows():
                                                col1, col2 = st.columns([3, 1])
                                                with col1:
                                                    # Reshape the vector back to 2D
                                                    sim_window = row["vector"].reshape(
                                                        store.window_size, 4
                                                    )
                                                    sim_window_display = (
                                                        denormalize_window(sim_window)
                                                    )

                                                    # Plot the window
                                                    fig = plot_window(
                                                        sim_window_display,
                                                        f"Similar {j+1}: {row['symbol']} - {row['window_start'].strftime('%Y-%m-%d')}",
                                                        row["window_start"],
                                                    )
                                                    st.plotly_chart(
                                                        fig, use_container_width=True
                                                    )

                                                with col2:
                                                    # Display metadata
                                                    meta = (
                                                        json.loads(row["metadata"])
                                                        if "metadata" in row
                                                        else {}
                                                    )
                                                    meta_str = "\n".join(
                                                        [
                                                            f"{k}: {v}"
                                                            for k, v in meta.items()
                                                            if k not in ["sample"]
                                                        ]
                                                    )
                                                    meta_str += f"\nDistance: {row.get('distance', 'N/A')}"
                                                    st.text_area(
                                                        f"Metadata Similar {j+1}",
                                                        meta_str,
                                                        height=300,
                                                    )

        else:
            # Mode 2: Compare to recent data
            st.header("Compare to Recent Data")

            col1, col2 = st.columns(2)
            with col1:
                # Select a symbol
                symbol = st.selectbox("Select symbol:", symbols)

                # Days back for recent window
                days_back = st.slider("Days to look back for recent window:", 1, 30, 7)

            with col2:
                # Number of similar windows to find
                n_similar = st.slider("Number of similar windows to find:", 1, 20, 5)

                # Exclude self or not
                exclude_self = st.checkbox(
                    "Exclude this symbol from results", value=True
                )

            # Button to find similar windows
            if st.button("Find Similar to Recent"):
                with st.spinner(f"Finding windows similar to recent {symbol} data..."):
                    similar_df = store.find_similar_to_recent(
                        symbol=symbol,
                        n=n_similar,
                        days_back=days_back,
                        exclude_self=exclude_self,
                    )

                    if similar_df.empty:
                        st.error(
                            f"No recent data or similar windows found for {symbol}."
                        )
                        return

                    # Display the query window (most recent)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_back)
                    windows, starts, metadata = store.get_windows(
                        symbol=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        limit=1,
                    )

                    if len(windows) > 0:
                        st.subheader(f"Most Recent Window for {symbol}")

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            window_display = denormalize_window(windows[0])
                            fig = plot_window(
                                window_display,
                                f"Recent: {starts[0].strftime('%Y-%m-%d')}",
                                starts[0],
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            meta_str = "\n".join(
                                [
                                    f"{k}: {v}"
                                    for k, v in metadata[0].items()
                                    if k not in ["sample"]
                                ]
                            )
                            st.text_area("Metadata (Recent)", meta_str, height=300)

                    # Display similar windows
                    st.subheader(f"Windows Similar to Recent {symbol}")

                    # Create a container for similar windows
                    similar_container = st.container()
                    with similar_container:
                        for j, row in similar_df.iterrows():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Reshape the vector back to 2D
                                sim_window = row["vector"].reshape(store.window_size, 4)
                                sim_window_display = denormalize_window(sim_window)

                                # Plot the window
                                fig = plot_window(
                                    sim_window_display,
                                    f"Similar {j+1}: {row['symbol']} - {row['window_start'].strftime('%Y-%m-%d')}",
                                    row["window_start"],
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                # Display metadata
                                meta = (
                                    json.loads(row["metadata"])
                                    if "metadata" in row
                                    else {}
                                )
                                meta_str = "\n".join(
                                    [
                                        f"{k}: {v}"
                                        for k, v in meta.items()
                                        if k not in ["sample"]
                                    ]
                                )
                                meta_str += f"\nDistance: {row.get('distance', 'N/A')}"
                                st.text_area(
                                    f"Metadata Similar {j+1}", meta_str, height=300
                                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()

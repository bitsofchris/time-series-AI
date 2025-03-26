"""
Page for finding windows similar to a symbol's recent window.
"""

import streamlit as st
from utils.db_connector import get_embedding_store, get_available_symbols
from components.visualizations import plot_window, plot_multiple_windows
import pandas as pd


def render():
    """Render the 'Find Similar to Recent' page."""
    st.header("Find Windows Similar to Recent Data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        symbols = get_available_symbols()
        default_idx = 0 if symbols else None
        source_symbol = st.selectbox("Source Symbol", symbols, index=default_idx)

    with col2:
        limit = st.slider("Number of Results", min_value=1, max_value=20, value=5)

    with col3:
        exclude_self = st.checkbox("Exclude Source Symbol", value=False)

    with col4:
        search_button = st.button("Find Similar to Recent")

    if not symbols:
        st.warning("No symbols found in the database. Please process some data first.")
        return

    if search_button and source_symbol:
        store = get_embedding_store()

        # First, get the most recent window that we'll be comparing to
        recent_window, recent_start, recent_metadata = store.get_most_recent_window(
            source_symbol
        )

        if recent_window is None:
            st.error(f"No recent window found for {source_symbol}")
            return

        # Display the query window (most recent window of the source symbol)
        st.subheader(f"Query Window: Most Recent {source_symbol}")

        # Generate dates for x-axis
        window_size = len(recent_window)
        timeframe = "1day"  # Default

        if recent_metadata and "timeframe" in recent_metadata:
            timeframe = recent_metadata["timeframe"]

        # Generate dates array
        if timeframe.endswith("day") or timeframe == "1d":
            # Daily data
            dates = [recent_start + pd.Timedelta(days=i) for i in range(window_size)]
        elif timeframe.endswith("hour") or timeframe.endswith("h"):
            # Hourly data
            dates = [recent_start + pd.Timedelta(hours=i) for i in range(window_size)]
        else:
            # Default to daily
            dates = [recent_start + pd.Timedelta(days=i) for i in range(window_size)]

        # Display chart
        st.plotly_chart(
            plot_window(
                recent_window,
                title=f"Query: {source_symbol} (Start: {recent_start.strftime('%Y-%m-%d')})",
                dates=dates,
            ),
            use_container_width=True,
        )

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

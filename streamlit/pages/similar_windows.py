"""
Page for finding the most similar windows to a selected window.
"""

import streamlit as st
from utils.db_connector import get_embedding_store, get_available_symbols
from components.visualizations import plot_window, plot_multiple_windows


def render():
    """Render the 'Find Most Similar Windows' page."""
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

"""
Page for finding windows similar to a symbol's recent window.
"""

import streamlit as st
from utils.db_connector import get_embedding_store, get_available_symbols
from components.visualizations import plot_multiple_windows


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

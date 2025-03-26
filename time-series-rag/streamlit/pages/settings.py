"""
Page for app settings.
"""

import streamlit as st
from database.connection import get_connection_string


def render():
    """Render the settings page."""
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

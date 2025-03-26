"""
Sidebar components for the Streamlit app.
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar and return the selected app mode."""
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Find Most Similar Windows", "Find Similar to Recent", "Settings"],
    )

    # Display app info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This app demonstrates time series similarity search using vector embeddings. "
        "Normalized price data is stored in LanceDB and can be queried for similarity."
    )

    return app_mode

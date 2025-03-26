"""
Configuration settings for the Streamlit app.
"""

import os
import sys
import streamlit as st
from datetime import datetime, timedelta

# Add src directory to Python path for importing project modules
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)


def configure_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="Time Series Similarity Search",
        page_icon="📊",
        layout="wide",
    )
    st.title("Time Series Similarity Search")


def get_default_settings():
    """Get default settings for the application."""
    from database.connection import get_connection_string

    return {
        "db_url": get_connection_string(),
        "lancedb_uri": "/Users/chris/repos/time-series-AI/lance-data/lancedb",
        "window_size": 20,
        "stride": 10,
    }


def initialize_session_state():
    """Initialize session state with default values if not already set."""
    defaults = get_default_settings()

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

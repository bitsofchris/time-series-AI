"""
Main entry point for the Time Series Similarity Search Streamlit app.
"""

import streamlit as st
from config import configure_page
from components.sidebar import render_sidebar
from pages import similar_windows, similar_to_recent, settings


def main():
    """Main function for the Streamlit app."""
    configure_page()

    # App mode selection from sidebar
    app_mode = render_sidebar()

    # Run the selected app mode
    if app_mode == "Find Most Similar Windows":
        similar_windows.render()
    elif app_mode == "Find Similar to Recent":
        similar_to_recent.render()
    elif app_mode == "Settings":
        settings.render()


if __name__ == "__main__":
    main()

"""
Visualization components for time series data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


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

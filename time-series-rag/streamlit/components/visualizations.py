"""
Visualization components for time series data.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta


def plot_window(window, title=None, height=300, dates=None):
    """
    Plot a normalized time series window as a candlestick chart.

    Args:
        window: 2D array of shape (window_size, 4) with normalized OHLC data
        title: Optional title for the plot
        height: Height of the plot in pixels
        dates: Optional list of datetime objects for x-axis. If None, uses timesteps.
    """
    # Create figure
    fig = go.Figure()

    # Extract OHLC data
    open_data = window[:, 0]
    high_data = window[:, 1]
    low_data = window[:, 2]
    close_data = window[:, 3]

    # Generate x-axis values (dates or time steps)
    if dates and len(dates) == len(open_data):
        x_values = dates
        x_title = "Date"
    else:
        x_values = list(range(len(open_data)))
        x_title = "Time Step"

    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=x_values,
            open=open_data,
            high=high_data,
            low=low_data,
            close=close_data,
            name="OHLC",
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )

    # Calculate and add volume if available (assuming it might be the 5th column)
    if window.shape[1] > 4:
        volume_data = window[:, 4]
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=volume_data,
                name="Volume",
                marker_color="rgba(100, 100, 250, 0.5)",
                yaxis="y2",
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title=x_title,
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,  # Trading view style has no range slider by default
        template="plotly_white",  # Clean white background similar to TradingView
    )

    # Add volume axis if needed
    if window.shape[1] > 4:
        fig.update_layout(
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False)
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

        # Get window dates for x-axis
        # Generate dates array using window_start and window size from metadata
        start_date = row["window_start"]
        if "parsed_metadata" in row and "timeframe" in row["parsed_metadata"]:
            timeframe = row["parsed_metadata"].get("timeframe", "1day")
        else:
            timeframe = "1day"

        # Generate sequence of dates based on timeframe
        if timeframe.endswith("day") or timeframe == "1d":
            # Daily data
            dates = [start_date + pd.Timedelta(days=i) for i in range(len(window))]
        elif timeframe.endswith("hour") or timeframe.endswith("h"):
            # Hourly data
            dates = [start_date + pd.Timedelta(hours=i) for i in range(len(window))]
        else:
            # Default to daily
            dates = [start_date + pd.Timedelta(days=i) for i in range(len(window))]

        col1, col2 = st.columns([3, 1])

        with col1:
            st.plotly_chart(
                plot_window(
                    window,
                    title=f"{symbol} (Start: {start_date.strftime('%Y-%m-%d')}, Distance: {distance:.4f})",
                    dates=dates,
                ),
                use_container_width=True,
            )

        with col2:
            # Display only the key metadata we care about
            st.subheader("Metadata")
            st.text(f"Symbol: {symbol}")

            # Timeframe from metadata if available
            if "parsed_metadata" in row and "timeframe" in row["parsed_metadata"]:
                st.text(f"Timeframe: {row['parsed_metadata']['timeframe']}")

            # Similarity score
            st.text(f"Similarity Score: {1.0 - distance:.4f}")
            st.text(f"Distance: {distance:.4f}")

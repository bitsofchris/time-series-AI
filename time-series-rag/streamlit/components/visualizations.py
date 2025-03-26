"""
Visualization components for time series data.
"""

import streamlit as st
import plotly.graph_objects as go


def plot_window(window, title=None, height=300):
    """
    Plot a normalized time series window as a candlestick chart.

    Args:
        window: 2D array of shape (window_size, 4) with normalized OHLC data
        title: Optional title for the plot
        height: Height of the plot in pixels
    """
    # Create figure
    fig = go.Figure()

    # Extract OHLC data
    open_data = window[:, 0]
    high_data = window[:, 1]
    low_data = window[:, 2]
    close_data = window[:, 3]

    # Generate sequential x-axis values (time steps)
    x_values = list(range(len(open_data)))

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
        xaxis_title="Time Step",
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

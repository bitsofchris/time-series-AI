# Time Series AI

Adventures in use time series data for fun and profit.


# Time Series RAG for Trading: Project Context and Implementation Plan

## Project Background

I'm a staff engineer with a background in time series research and 7 years of previous day trading experience. I'm looking to build a prototype over the next 3 weeks that leverages both my trading background and current technical expertise. The project aims to create a time series pattern matching system for trading.

## Project Concept

The core idea is to build a "Time Series RAG" (Retrieval Augmented Generation) system that can:
1. Identify similar historical chart patterns to current market conditions
2. Allow exploration of what typically happened after these similar patterns occurred
3. Potentially provide trading insights based on pattern recognition

The system will use embedding techniques to convert time series data (price charts) into vector representations, enabling similarity search across different timeframes and symbols.

## Technical Approach

### Data Pipeline
- Collect historical market data for multiple symbols and timeframes
- Store raw data in TimescaleDB (time-series optimized PostgreSQL)
- Process raw data into normalized forms suitable for pattern comparison
- Generate overlapping windows to capture patterns of various lengths
- Create embeddings for each window that capture the essential characteristics of the price action

### Similarity Search
- Use vector database (LanceDB) to store embeddings with metadata
- Implement efficient similarity search across all historical patterns
- Support timeframe-agnostic matching (e.g., hourly patterns can match daily patterns)
- Allow filtering by metadata (symbol, date range, etc.)

### Visualization
- Build Streamlit app with interactive candlestick charts
- Show current market pattern alongside historical similar patterns
- Display statistical analysis of outcomes following similar patterns
- Allow interactive exploration of various symbols and timeframes

## Implementation Plan

### Week 1: Foundation
- [x] Set up project repository with clear structure
- [x] Implement TimescaleDB in Docker with migration-based schema management
- [ ] Create data ingestion pipeline using yfinance (and potentially CCXT for crypto)
- [ ] Build initial data normalization and processing logic
- [ ] Develop basic Streamlit visualization for data exploration

### Week 2: Core Functionality
- [ ] Research and implement time series embedding generation
- [ ] Set up vector database for pattern storage and retrieval
- [ ] Build similarity search functionality
- [ ] Create pattern comparison visualization components
- [ ] Implement window generation and feature extraction

### Week 3: Refinement and Analysis
- [ ] Enhance UI for intuitive pattern exploration
- [ ] Add statistical analysis of historical outcomes
- [ ] Implement backtesting of pattern-based signals
- [ ] Optimize performance for responsive user experience
- [ ] Document findings and evaluate prototype effectiveness

## Technical Stack

- **Database:** TimescaleDB (for raw time series) + LanceDB (for embeddings)
- **Backend:** Python with SQLAlchemy Core
- **Data Processing:** Pandas, NumPy
- **Visualization:** Streamlit with Plotly
- **Data Sources:** yfinance (stocks), potentially CCXT (crypto)
- **Deployment:** Local Docker environment

## Key Research Questions

1. What embedding approaches best capture meaningful market patterns?
2. How to normalize patterns across different timeframes effectively?
3. What window sizes and overlap strategies work best for pattern recognition?
4. How predictive are visually similar patterns of future market movements?
5. What additional features beyond price/volume improve pattern matching?

## Success Criteria

- Functional end-to-end prototype that can find visually similar chart patterns
- Support for timeframe-agnostic pattern matching
- Interactive UI for pattern exploration and comparison
- Basic statistical analysis of pattern outcomes
- Assessment of whether the approach shows potential for trading edge

This project leverages my unique combination of trading experience and technical skills while aligning with my current work in time series research, creating a sustainable path to potentially valuable trading tools.

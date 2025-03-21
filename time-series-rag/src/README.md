# Time Series RAG - Source Code

This directory contains the source code for the Time Series RAG project.

## Data Fetching

The data fetching module in `data/fetchers` provides functionality to download market data from various sources and store it in TimescaleDB.

### YFinance Fetcher

The `yfinance_fetcher.py` module fetches data from Yahoo Finance and stores it directly in TimescaleDB in both raw and normalized forms.

#### Running from the Command Line

You can use the `fetch_market_data.py` script to download data:

```bash
# Navigate to the src directory
cd time-series-rag/src

# Fetch 1 year of daily data for Apple, Microsoft, and Google
python scripts/fetch_market_data.py --symbols AAPL,MSFT,GOOGL --timeframe 1d --days 365

# Fetch 30 days of hourly data
python scripts/fetch_market_data.py --symbols AAPL,MSFT,GOOGL --timeframe 1h --days 30

# Skip normalization
python scripts/fetch_market_data.py --symbols AAPL --timeframe 1d --no-normalize

# Specify a custom database URL
python scripts/fetch_market_data.py --symbols AAPL --db-url postgresql://user:pass@host:port/dbname
```

#### Using in Python Code

You can also use the fetcher directly in your Python code:

```python
from data.fetchers.yfinance_fetcher import YFinanceFetcher
from datetime import datetime, timedelta

# Initialize fetcher
fetcher = YFinanceFetcher()

# Define parameters
symbols = ['AAPL', 'MSFT', 'GOOGL']
timeframe = '1d'
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()

# Fetch and store data
results = fetcher.fetch_and_store(
    symbols=symbols,
    timeframe=timeframe,
    start_date=start_date,
    end_date=end_date,
)

print(f"Total rows inserted: {sum(results.values())}")
```

## Database Structure

The data is stored in two main tables:

1. `raw_price_data` - Contains raw price data exactly as received from the source
2. `normalized_price_data` - Contains normalized price data for pattern matching

Both tables are TimescaleDB hypertables optimized for time-series queries.

## Adding More Data Sources

To add support for more data sources:

1. Create a new fetcher module in `data/fetchers/` folder
2. Implement a similar interface to the YFinanceFetcher
3. Create a command-line script in `scripts/` folder to use the new fetcher 
# Docker Setup for Time Series RAG

This directory contains Docker configuration for running the Time Series RAG project.

## Components

1. **TimescaleDB**: Time-series optimized PostgreSQL database for storing market data
2. **LanceDB**: Vector database for storing embeddings and similarity search
3. **Streamlit**: Web application for visualizing and exploring data

## Quick Start

To run the entire stack with a single command:

```bash
# Navigate to the docker directory
cd time-series-rag/docker

# Start all services
docker-compose up -d
```

This will start all required services in the background.

## Accessing Services

- **TimescaleDB**: Available at `localhost:5432`
  - Username: `timeseriesrag`
  - Password: `devpassword`
  - Database: `market_data`

- **LanceDB**: Available at `localhost:8000`

- **Streamlit App**: Available at `localhost:8501`

## Loading Data

After starting the services, you can load market data using the provided script:

```bash
# Navigate to the project root
cd time-series-rag

# Make sure the DATABASE_URL environment variable is set
export DATABASE_URL="postgresql://timeseriesrag:devpassword@localhost:5432/market_data"

# Run the data fetcher script
python src/scripts/fetch_market_data.py --symbols AAPL,MSFT,GOOGL --timeframe 1d
```

## Common Tasks

### Running Database Migrations

The database migrations are automatically applied when the container is first created.
For new migrations, you can use the dedicated service:

```bash
docker-compose run migrate
```

### Monitoring Logs

```bash
# View logs for all services
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f timescaledb
```

### Stopping Services

```bash
docker-compose down
```

### Removing Data (Reset)

```bash
docker-compose down -v
```

This command removes all volumes along with containers, effectively resetting all data.

## Customization

You can customize the Docker setup by editing:

- `docker-compose.yml`: Service configuration
- `Dockerfile`: Application build instructions
- Environment variables in `docker-compose.yml` 
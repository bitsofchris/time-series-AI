-- Migration: 001_initial_tables.sql
-- Basic schema for time series RAG project
-- Created: 2024-03-20

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Raw price data table
CREATE TABLE IF NOT EXISTS raw_price_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL, 
    timeframe VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('raw_price_data', 'time');

-- Normalized price data for pattern matching
CREATE TABLE IF NOT EXISTS normalized_price_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open_norm DOUBLE PRECISION NOT NULL,
    high_norm DOUBLE PRECISION NOT NULL,
    low_norm DOUBLE PRECISION NOT NULL,
    close_norm DOUBLE PRECISION NOT NULL,
    volume_norm DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('normalized_price_data', 'time');

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_raw_price_data_symbol ON raw_price_data(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_normalized_price_symbol ON normalized_price_data(symbol, timeframe);
"""
Configuration settings for the Quantitative Trading Strategy Backtester.
Contains default parameters, asset tickers, and system settings.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any

# =============================================================================
# ASSET UNIVERSE
# =============================================================================

# Stock tickers (Large Cap Tech + Diversified)
STOCK_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",  # Big Tech
    "NVDA", "TSLA", "JPM", "V", "JNJ",         # Diversified sectors
    "WMT", "PG", "UNH", "HD", "BAC"            # Consumer & Finance
]

# ETF tickers
ETF_TICKERS: List[str] = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000
    "EFA",   # International Developed
    "EEM",   # Emerging Markets
    "GLD",   # Gold
    "TLT",   # Long-term Treasury
    "XLF",   # Financials
    "XLE",   # Energy
    "XLK"    # Technology
]

# Cryptocurrency tickers (Yahoo Finance format)
CRYPTO_TICKERS: List[str] = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
    "ADA-USD",  # Cardano
    "XRP-USD"   # Ripple
]

# All tickers combined
ALL_TICKERS: List[str] = STOCK_TICKERS + ETF_TICKERS + CRYPTO_TICKERS

# =============================================================================
# TIME SETTINGS
# =============================================================================

DEFAULT_START_DATE: str = (datetime.now() - timedelta(days=365*7)).strftime("%Y-%m-%d")
DEFAULT_END_DATE: str = datetime.now().strftime("%Y-%m-%d")
DATA_FREQUENCY: str = "1d"  # Daily data

# =============================================================================
# TRANSACTION COSTS
# =============================================================================

TRANSACTION_COSTS: Dict[str, float] = {
    "fixed_fee": 0.0,           # Fixed fee per trade ($)
    "percentage_fee": 0.001,    # 0.1% commission (10 bps)
    "slippage_pct": 0.0005,     # 0.05% slippage (5 bps)
}

# =============================================================================
# RISK MANAGEMENT DEFAULTS
# =============================================================================

RISK_PARAMS: Dict[str, Any] = {
    "stop_loss_pct": 0.05,           # 5% stop loss
    "take_profit_pct": 0.15,         # 15% take profit
    "max_position_size": 0.20,       # Max 20% in single position
    "min_position_size": 0.02,       # Min 2% position
    "use_volatility_sizing": True,   # Size based on volatility
    "target_volatility": 0.15,       # 15% annual target vol
    "use_kelly_criterion": False,    # Kelly criterion sizing
    "kelly_fraction": 0.25,          # Fractional Kelly (conservative)
}

# =============================================================================
# STRATEGY DEFAULTS
# =============================================================================

MOMENTUM_PARAMS: Dict[str, Any] = {
    "lookback_period": 20,      # Days to look back
    "holding_period": 5,        # Days to hold position
    "top_n": 5,                 # Number of top performers to buy
    "threshold": 0.0,           # Minimum return threshold
}

MEAN_REVERSION_PARAMS: Dict[str, Any] = {
    "lookback_period": 20,      # MA lookback
    "entry_zscore": -2.0,       # Z-score to enter (buy when low)
    "exit_zscore": 0.0,         # Z-score to exit
    "bollinger_std": 2.0,       # Bollinger band std dev
}

PAIRS_TRADING_PARAMS: Dict[str, Any] = {
    "lookback_period": 60,      # Cointegration lookback
    "entry_zscore": 2.0,        # Entry threshold
    "exit_zscore": 0.5,         # Exit threshold
    "correlation_threshold": 0.7,  # Min correlation for pair
}

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================

BACKTEST_PARAMS: Dict[str, Any] = {
    "initial_capital": 100000,  # Starting capital ($)
    "rebalance_frequency": "weekly",  # daily, weekly, monthly
    "benchmark": "SPY",         # Benchmark for comparison
}

# =============================================================================
# MONTE CARLO SETTINGS
# =============================================================================

MONTE_CARLO_PARAMS: Dict[str, Any] = {
    "n_simulations": 1000,      # Number of simulations
    "return_perturbation": 0.001,  # Random noise to add
    "bootstrap_block_size": 20,    # Block bootstrap size
    "confidence_levels": [0.05, 0.25, 0.50, 0.75, 0.95],
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

PLOT_SETTINGS: Dict[str, Any] = {
    "figure_width": 12,
    "figure_height": 6,
    "color_scheme": {
        "equity": "#2E86AB",
        "benchmark": "#A23B72",
        "drawdown": "#F18F01",
        "positive": "#4CAF50",
        "negative": "#F44336",
    },
    "style": "seaborn-v0_8-whitegrid",
}

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

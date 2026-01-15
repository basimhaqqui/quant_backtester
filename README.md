# Quantitative Trading Strategy Backtester

A comprehensive, production-ready backtesting framework for quantitative trading strategies. Built with Python, this system allows you to test momentum, mean-reversion, and pairs trading strategies on historical market data with realistic transaction costs, slippage, and risk management.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

### Core Functionality
- **Multiple Trading Strategies**
  - Momentum: Cross-sectional momentum with configurable lookback periods
  - Mean Reversion: Bollinger Bands and z-score based entry/exit
  - Pairs Trading: Statistical arbitrage using cointegration analysis
  - Dual Momentum: Combines absolute and relative momentum filters

### Realistic Simulation
- **Transaction Cost Modeling**: Fixed fees + percentage-based commissions
- **Slippage Simulation**: Market impact modeling for realistic fills
- **Risk Management**: Stop-loss, take-profit, and position sizing
- **Position Sizing**: Volatility-based and Kelly criterion support

### Analytics & Reporting
- **Performance Metrics**: Sharpe, Sortino, CAGR, Max Drawdown, Win Rate, Profit Factor
- **Monte Carlo Simulation**: 1000+ iterations for strategy robustness testing
- **Interactive Dashboard**: Streamlit-based UI for parameter exploration
- **PDF Reports**: Professional reports with charts and analysis

### Data Support
- **Multi-Asset Classes**: Stocks, ETFs, and Cryptocurrencies
- **Data Sources**: Yahoo Finance (yfinance)
- **Historical Range**: 5-10 years of daily data supported

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quant_backtester.git
cd quant_backtester

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Quick demo with sample data and momentum strategy
python main.py demo
```

### Launch Dashboard

```bash
# Start the interactive Streamlit dashboard
python main.py dashboard
# Then open http://localhost:8501 in your browser
```

### Run Custom Backtest

```bash
# Run momentum strategy on specific tickers
python main.py backtest \
    --strategy momentum \
    --tickers AAPL MSFT GOOGL AMZN META \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --capital 100000 \
    --report \
    --monte-carlo

# Run mean reversion strategy
python main.py backtest \
    --strategy mean_reversion \
    --tickers SPY QQQ IWM \
    --entry-zscore -2.0 \
    --lookback 20 \
    --report
```

## Project Structure

```
quant_backtester/
├── config.py              # Configuration and default parameters
├── data_fetcher.py        # Market data retrieval (yfinance)
├── backtester.py          # Core backtesting engine
├── metrics.py             # Performance metrics calculation
├── risk_management.py     # Position sizing, stop-loss, etc.
├── monte_carlo.py         # Monte Carlo simulation
├── visualizations.py      # Charts and plots (Matplotlib/Plotly)
├── report_generator.py    # PDF report generation
├── app.py                 # Streamlit dashboard
├── main.py                # CLI entry point
├── strategies/
│   ├── __init__.py
│   ├── base.py            # Base strategy class
│   ├── momentum.py        # Momentum strategies
│   ├── mean_reversion.py  # Mean reversion strategies
│   └── pairs_trading.py   # Pairs trading strategy
├── requirements.txt
└── README.md
```

## Usage Examples

### Python API

```python
from data_fetcher import DataFetcher
from backtester import Backtester
from strategies.momentum import MomentumStrategy
from risk_management import RiskManager

# Fetch historical data
fetcher = DataFetcher()
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "SPY"]
prices = fetcher.fetch_multiple_tickers(tickers, "2020-01-01", "2024-01-01")

# Create strategy
strategy = MomentumStrategy(
    lookback_period=20,
    holding_period=5,
    top_n=3
)

# Configure risk management
risk_manager = RiskManager(
    stop_loss_pct=0.05,      # 5% stop loss
    take_profit_pct=0.15,    # 15% take profit
    max_position_size=0.20   # Max 20% per position
)

# Run backtest
backtester = Backtester(
    initial_capital=100000,
    commission_pct=0.001,    # 10 bps
    slippage_pct=0.0005,     # 5 bps
    risk_manager=risk_manager
)

result = backtester.run(strategy, prices)

# View results
print(result.metrics.summary())
print(f"Total Return: {result.metrics.total_return*100:.1f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
```

### Monte Carlo Simulation

```python
from monte_carlo import MonteCarloSimulator

mc = MonteCarloSimulator(n_simulations=1000)
mc_result = mc.run_simulation(strategy, prices)

print(mc_result.summary())

# Access confidence intervals
print(f"95% CI for Return: {mc_result.confidence_intervals['return']['5%']*100:.1f}% to {mc_result.confidence_intervals['return']['95%']*100:.1f}%")
```

### Generate PDF Report

```python
from report_generator import ReportGenerator

generator = ReportGenerator(output_dir="./reports")
report_path = generator.generate_report(result, mc_result)
print(f"Report saved: {report_path}")
```

## Strategy Details

### Momentum Strategy
Buys assets with the highest recent returns based on the momentum anomaly.

**Parameters:**
- `lookback_period`: Days to calculate momentum (default: 20)
- `holding_period`: Days to hold before rebalancing (default: 5)
- `top_n`: Number of top performers to buy (default: 5)
- `long_short`: Also short worst performers (default: False)

### Mean Reversion Strategy
Buys oversold assets (below Bollinger Bands) expecting price to revert to mean.

**Parameters:**
- `lookback_period`: MA period for Bollinger Bands (default: 20)
- `entry_zscore`: Z-score threshold for entry (default: -2.0)
- `exit_zscore`: Z-score threshold for exit (default: 0.0)
- `use_rsi_filter`: Confirm with RSI indicator (default: True)

### Pairs Trading Strategy
Trades spread between cointegrated asset pairs.

**Parameters:**
- `lookback_period`: Period for cointegration analysis (default: 60)
- `entry_zscore`: Spread z-score for entry (default: 2.0)
- `exit_zscore`: Spread z-score for exit (default: 0.5)
- `correlation_threshold`: Min correlation for pair selection (default: 0.7)

## Performance Metrics

The backtester calculates comprehensive performance metrics:

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return over backtest period |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Risk-adjusted return (excess return / volatility) |
| Sortino Ratio | Downside risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Calmar Ratio | CAGR / Max Drawdown |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profits / Gross losses |
| VaR (95%) | Value at Risk at 95% confidence |
| CVaR (95%) | Conditional VaR (Expected Shortfall) |

## Sample Results

Example backtest results for Momentum strategy (2020-2024):

```
==================================================
PERFORMANCE SUMMARY
==================================================
Total Return:          87.34%
CAGR:                  17.12%
Sharpe Ratio:           1.24
Sortino Ratio:          1.87
Max Drawdown:         -18.45%
Volatility:            15.23%
--------------------------------------------------
Win Rate:              54.32%
Profit Factor:          1.43
Total Trades:           156
--------------------------------------------------
VaR (95%):             -2.12%
CVaR (95%):            -3.01%
==================================================
```

## Customization & Extension

### Adding New Strategies

1. Create a new file in `strategies/` directory
2. Inherit from `Strategy` base class
3. Implement `generate_signals()` and `get_required_history()`

```python
from strategies.base import Strategy, Signal

class MyCustomStrategy(Strategy):
    def __init__(self, param1=10, param2=0.5):
        super().__init__(name="MyStrategy", params={"param1": param1, "param2": param2})
        self.param1 = param1
        self.param2 = param2

    def get_required_history(self) -> int:
        return self.param1 + 5

    def generate_signals(self, prices, current_date, positions=None):
        signals = []
        # Your signal generation logic here
        return signals
```

### Integration with Portfolio Optimizer

This backtester can be integrated with mean-variance optimization:

```python
from scipy.optimize import minimize
import numpy as np

def optimize_weights(returns, risk_aversion=1):
    """Mean-variance optimization for strategy weights."""
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    def objective(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - risk_aversion * port_vol**2)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(objective, n_assets * [1/n_assets],
                     method="SLSQP", bounds=bounds, constraints=constraints)

    return result.x
```

## Tech Stack

- **Python 3.9+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Statistical analysis
- **yfinance** - Market data API
- **Matplotlib/Plotly** - Visualization
- **Streamlit** - Interactive dashboard
- **fpdf2** - PDF generation

## Future Enhancements

- [ ] Options strategies (covered calls, protective puts)
- [ ] Machine learning signal generation (LSTM, Random Forest)
- [ ] Real-time paper trading mode
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization integration
- [ ] Alpaca/Interactive Brokers live trading
- [ ] Database storage for backtests
- [ ] Parallel backtesting for parameter sweeps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

## Author

Built with Python for quantitative finance research and portfolio analysis.

---

**Happy Backtesting!** 📈

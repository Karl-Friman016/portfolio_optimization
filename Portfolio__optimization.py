"""
Portfolio Optimization Project
Author: Karl Friman
Date: 2026-03-10

Purpose:
This project uses historical stock price data to analyze portfolio returns and risk.
The goal is to build and compare portfolios using Modern Portfolio Theory, including
an equally weighted portfolio, the minimum variance portfolio and the portfolio with the highest Sharpe ratio.

Financial idea:
A portfolio's expected return depends on the weighted average of asset returns,
while portfolio risk depends on both individual asset volatility and correlations
between assets. By changing the portfolio weights, we can analyze the trade-off
between risk and return.

Steps in this script:
1. Import libraries
2. Set project parameters
3. Download historical price data
4. Calculate daily returns
5. Estimate expected returns and covariance matrix
6. Calculate equally weighted portfolio performance
7. Simulate random portfolios
8. The portfolio with the highest Sharpe ratio
9. Minimum variance portfolio

Part 2 Mean variance optimization
10. Set target returns for the efficient frontier
11. Compute the efficient frontier
12. plot the prortfolios
13. Comparison tabel 

Notes:
- This project is for personal learning
- Expected returns are estimated using historical average returns.
"""
#%%
# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

#%%
# --------------------------------------------------
# 2. Set parameters
# --------------------------------------------------
tickers = ['MSFT', 'AAPL', 'KO', 'DE', 'JNJ', 'PEP', 'CAT', 'MRK']
start_date = '2010-01-01'
end_date = '2025-12-31'
trading_days = 252
num_portfolios = 3000
risk_free_rate = 0.015

#%%
# --------------------------------------------------
# 3. Download historical price data
# --------------------------------------------------
price_data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False
)

prices = price_data['Adj Close']

#%%
# --------------------------------------------------
# 4. Calculate daily returns
# --------------------------------------------------
returns = prices.pct_change().dropna()

#%%
# --------------------------------------------------
# 5. Estimate expected returns and covariance matrix
# --------------------------------------------------
expected_returns = returns.mean() * trading_days
cov_matrix = returns.cov() * trading_days

#%%
# --------------------------------------------------
# 6. Calculate equally weighted portfolio performance
# --------------------------------------------------
num_assets = len(expected_returns)
equal_weights = np.ones(num_assets) / num_assets

equal_return = np.dot(equal_weights, expected_returns)
equal_variance = np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights))
equal_volatility = np.sqrt(equal_variance)
equal_sharpe = (equal_return - risk_free_rate) / equal_volatility

print("Equally weighted portfolio:")
for stock, weight in zip(expected_returns.index, equal_weights):
    print(f"{stock}: {weight:.2%}")

print(f"\nExpected annual return: {equal_return:.2%}")
print(f"Annual volatility: {equal_volatility:.2%}")
print(f"Sharpe ratio: {equal_sharpe:.2f}")

#%%
# --------------------------------------------------
# 7. Simulate random portfolios
# --------------------------------------------------
simulated_returns = []
simulated_volatilities = []
simulated_sharpes = []
simulated_weights = []

for _ in range(num_portfolios):
    # Generate random portfolio weights that sum to 1
    weights = np.random.random(num_assets)
    weights = weights / np.sum(weights)

    # Calculate portfolio performance
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Store results
    simulated_returns.append(portfolio_return)
    simulated_volatilities.append(portfolio_volatility)
    simulated_sharpes.append(portfolio_sharpe)
    simulated_weights.append(weights)


# Convert results to arrays
simulated_returns = np.array(simulated_returns)
simulated_volatilities = np.array(simulated_volatilities)
simulated_sharpes = np.array(simulated_sharpes)

#%%
# --------------------------------------------------
# 8. The portfolio with the highest Sharpe ratio
# --------------------------------------------------
max_sharpe_idx = np.argmax(simulated_sharpes)

optimal_return = simulated_returns[max_sharpe_idx]
optimal_volatility = simulated_volatilities[max_sharpe_idx]
optimal_sharpe = simulated_sharpes[max_sharpe_idx]
optimal_weights = simulated_weights[max_sharpe_idx]

print("\nOptimal portfolio (maximum Sharpe ratio):")
for stock, weight in zip(expected_returns.index, optimal_weights):
    print(f"{stock}: {weight:.2%}")

print(f"\nExpected annual return: {optimal_return:.2%}")
print(f"Annual volatility: {optimal_volatility:.2%}")
print(f"Sharpe ratio: {optimal_sharpe:.2f}")

#%%
#===================================================
# 9. The minimum variance portfolio
#===================================================

# Find index of portfolio with the lowest volatility
min_vol_idx = np.argmin(simulated_volatilities)

# Extract the performance and weights of the minimum variance portfolio
min_var_return = simulated_returns[min_vol_idx]
min_var_volatility = simulated_volatilities[min_vol_idx]
min_var_sharpe = simulated_sharpes[min_vol_idx]
min_var_weights = simulated_weights[min_vol_idx]

# Display the minimum variance portfolio
print("\nMinimum variance portfolio:")
for stock, weight in zip(expected_returns.index, min_var_weights):
    print(f"{stock}: {weight:.2%}")

print(f"\nExpected annual return: {min_var_return:.2%}")
print(f"Annual volatility: {min_var_volatility:.2%}")
print(f"Sharpe ratio: {min_var_sharpe:.2f}")

# %%
#=============================================
# Part 2: Mean-variance Optimization
#=============================================

from scipy.optimize import minimize

def mv_portfolio_return(mv_weights, expected_returns):
    return np.dot(mv_weights, expected_returns)

def mv_portfolio_volatility(mv_weights, cov_matrix):
    mv_variance = np.dot(mv_weights.T, np.dot(mv_weights,cov_matrix))
    return np.sqrt(mv_variance)

#%%
#====================================================
# 10. Set target returns for the efficient frontier
#====================================================

mv_target_returns = np.linspace(
    simulated_returns.min(),
    simulated_returns.max(),
    50
)

mv_frontier_volatilities = []
mv_frontier_weights = []

#%%
#==================================================
# 11. Compute the efficient frontier
#==================================================

for target_return in mv_target_returns:
    # Constraint 1: Portfolio weights must sum to 1
    # Constraint 2: Portfolio return must equal target return
    constraints = (
        {'type': 'eq', 'fun': lambda mv_weights: np.sum(mv_weights) - 1},
        {'type': 'eq', 'fun': lambda mv_weights: mv_portfolio_return(mv_weights, expected_returns) - target_return}
    )
    
    # No short selling: each weight must stay between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Start optimization from equal weights
    initial_weights = np.ones(num_assets) / num_assets

    # Minimize portfolio volatility for the given target return
    result = minimize(
        mv_portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Store the optimized volatility and weights
    if result.success:
        mv_frontier_volatilities.append(result.fun)
        mv_frontier_weights.append(result.x)
    else:
        mv_frontier_volatilities.append(np.nan)
        mv_frontier_weights.append(None)

# Convert frontier results to an array
mv_frontier_volatilities = np.array(mv_frontier_volatilities)

#%%
# --------------------------------------------------
# 12. Plot simulated portfolios
# --------------------------------------------------
plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    simulated_volatilities,
    simulated_returns,
    c=simulated_sharpes,
    alpha=0.5
)

plt.scatter(
    optimal_volatility,
    optimal_return,
    marker='*',
    s=300,
    label='Maximum Sharpe Ratio'
)

plt.scatter(
    equal_volatility,
    equal_return,
    marker='o',
    s=150,
    label='Equal-Weighted Portfolio'
)

plt.scatter(
    min_var_volatility,
    min_var_return,
    marker='X',
    s=200,
    label='Minimum Variance'
)

plt.plot(
    mv_frontier_volatilities,
    mv_target_returns,
    linewidth=2,
    label='Efficient Frontier'
)

plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Annual Volatility')
plt.ylabel('Expected Annual Return')
plt.title('Simulated Portfolio Optimization')
plt.legend()
plt.show()

#%%
# --------------------------------------------------
# 13. Portfolio comparison table
# --------------------------------------------------

portfolio_summary = pd.DataFrame({
    'Portfolio': [
        'Equal Weight',
        'Max Sharpe',
        'Min Variance (Simulated)'
    ],
    'Expected Return': [
        equal_return,
        optimal_return,
        min_var_return
    ],
    'Volatility': [
        equal_volatility,
        optimal_volatility,
        min_var_volatility
    ],
    'Sharpe Ratio': [
        equal_sharpe,
        optimal_sharpe,
        min_var_sharpe
    ]
})

# Set portfolio names as index
portfolio_summary.set_index('Portfolio', inplace=True)

# Display table
print("\nPortfolio Comparison:")
print(portfolio_summary)
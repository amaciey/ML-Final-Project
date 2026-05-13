import pandas as pd
import numpy as np
import stockstats as ss

full_price_df = pd.read_csv('final_dataset.csv')

price_columns = ['HVO_class_II_fob_ARA', 'Ucome_fob_ARA', 'UCO_exw_ARA', 'EU_BRENT_CRUDE', 'LSMGO_Rotterdam']

indicators_list = ['SMA_', 'EMA_', 'LRMA_', 'KAMA_', 'CTI_', 'RSI_', 'MACD', 'TRIX', 'PPO', 'Stochastic_', 'RoC_',
                   'Coppock_', 'KST', 'PGO', 'PSL', 'MSTD_', 'MVAR_', 'BB_', 'KER_', 'DMA_', 'MAD_', 'BIAS_']

# Calculate technical indicators for each column
# Using all 22 indicators from README.md organized by category

for column in price_columns:
    print(f'Calculating Indicators for {column}')
    # Create a copy of the price data for stockstats
    price_data = full_price_df[[column]].copy()
    price_data.columns = ['close']
    
    # Initialize StockDataFrame
    stock = ss.StockDataFrame(price_data)
    
    # ===== TREND-FOLLOWING INDICATORS =====
    full_price_df[f'{column}_SMA_20'] = stock['close_20_sma']
    full_price_df[f'{column}_SMA_50'] = stock['close_50_sma']
    full_price_df[f'{column}_EMA_12'] = stock['close_12_ema']
    full_price_df[f'{column}_EMA_26'] = stock['close_26_ema']
    full_price_df[f'{column}_LRMA_20'] = stock['close_20_lrma']  # Linear Regression
    full_price_df[f'{column}_KAMA_10'] = stock['close_10_kama']  # Kaufman's Adaptive MA
    full_price_df[f'{column}_CTI_20'] = stock['close_20_cti']  # Correlation Trend Indicator
    
    # ===== MOMENTUM INDICATORS =====
    full_price_df[f'{column}_RSI_14'] = stock['rsi']  # Default 14-period
    full_price_df[f'{column}_MACD'] = stock['macd']
    full_price_df[f'{column}_MACD_Signal'] = stock['macds']
    full_price_df[f'{column}_MACD_Histogram'] = stock['macdh']
    full_price_df[f'{column}_TRIX'] = stock['trix']
    full_price_df[f'{column}_PPO'] = stock['ppo']
    full_price_df[f'{column}_Stochastic_RSI'] = stock['stochrsi']
    full_price_df[f'{column}_RoC_12'] = stock['close_12_roc']  # Rate of Change: 12-period
    full_price_df[f'{column}_Coppock_Curve'] = stock['coppock']  # Coppock Curve
    full_price_df[f'{column}_KST'] = stock['kst']  # Know Sure Thing
    full_price_df[f'{column}_PSL'] = stock['psl']  # Percentage Signal Line
    
    # ===== VOLATILITY INDICATORS =====
    full_price_df[f'{column}_MSTD_20'] = stock['close_20_mstd']  # Moving Standard Deviation
    full_price_df[f'{column}_MVAR_20'] = stock['close_20_mvar']  # Moving Variance
    full_price_df[f'{column}_BB_Upper'] = stock['boll_ub']  # Bollinger Band Upper
    full_price_df[f'{column}_BB_Lower'] = stock['boll_lb']  # Bollinger Band Lower
    full_price_df[f'{column}_BB_Middle'] = stock['boll']  # Bollinger Band Middle
    full_price_df[f'{column}_KER_20'] = stock['ker_20']  # Kaufman Efficiency Ratio
    
    # ===== OTHER INDICATORS =====
    full_price_df[f'{column}_DMA_20_50'] = stock['dma']  # Difference of Moving Averages
    full_price_df[f'{column}_MAD_20'] = stock['close_20_mad']  # Mean Absolute Deviation

print("\nAll 26 technical indicators calculated successfully!")
indicator_cols = [col for col in full_price_df.columns if any(ind in col for ind in indicators_list)]
print(f"\nIndicators added ({len(indicator_cols)}) total columns:")

print(f"\nDataFrame shape: {full_price_df.shape}")
full_price_df.to_csv('final_dataset_with_indicators.csv')
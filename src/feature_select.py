import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import datetime

def feature_selection(data):
    # Load dataset with indicators
    # data = pd.read_csv('./data/processed/final_dataset_with_indicators.csv')
    df = data.loc[:,~data.columns.isin(['Unnamed: 0'])]
    exclude = "Ucome"
    df = df.loc[:, ~df.columns.str.contains(exclude, case=False, na=False)]
    df = df.drop(columns=['Date'])
    #remove row with initial Nan values from indicators
    df = df.iloc[1:,:]
    # define X and y variables for feature selection
    X = df
    y = data['Ucome_fob_ARA'].iloc[1:]
    # 2. Initialize and train the Random Forest with 100 trees as a baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    # storing the importances in a pandas Series
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    # 4. Use SelectFromModel to filter features
    # This uses a threshold (e.g., 'median' or a specific number)
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_important = selector.transform(X)
    return X_important
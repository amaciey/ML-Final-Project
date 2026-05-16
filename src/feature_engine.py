import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_breast_cancer

# 1. Load a sample dataset (features = 30)
data = pd.read_csv('data/processed/final_dataset_with_indicators.csv')
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# 2. Initialize and train the Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 3. View the raw importance scores
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("Top 5 Features:")
print(importances.head(5))

# 4. Use SelectFromModel to filter features
# This uses a threshold (e.g., 'median' or a specific number)
selector = SelectFromModel(rf, threshold='median', prefit=True)

X_important = selector.transform(X)

print(f"\nOriginal shape: {X.shape}")
print(f"Reduced shape: {X_important.shape}")

# 5. Visualize the importance
importances.plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importance using Random Forest")
plt.show()
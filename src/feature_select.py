from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import pandas as pd

def _prepare_feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only candidate predictors for RF feature selection."""
    df = data.loc[:, ~data.columns.isin(["Unnamed: 0"])]
    df = df.loc[:, ~df.columns.str.contains("Ucome", case=False, na=False)]
    df = df.drop(columns=["Date"], errors="ignore")
    return df.iloc[1:, :]


def feature_selection(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame = None,
    threshold: str = "median",
):
    """
    Fit feature selector on train only and apply to train/test.

    Backward-compatible behavior:
    - If `test_data` is None, returns transformed train DataFrame.
    - If `test_data` is provided, returns (train_transformed, test_transformed, chosen_features).
    """
    X_train = _prepare_feature_matrix(train_data)
    y_train = train_data["Ucome_fob_ARA"].iloc[1:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)

    chosen_features = X_train.columns[selector.get_support()]
    X_train_selected = selector.transform(X_train)
    train_selected_df = pd.DataFrame(
        X_train_selected,
        columns=chosen_features,
        index=X_train.index,
    )
    train_selected_df["Ucome_fob_ARA"] = y_train.values
    train_selected_df = train_selected_df.reset_index(drop=True)

    if test_data is None:
        return train_selected_df

    X_test = _prepare_feature_matrix(test_data)
    y_test = test_data["Ucome_fob_ARA"].iloc[1:]

    # Ensure same column order before transform.
    X_test = X_test.reindex(columns=X_train.columns)
    X_test_selected = selector.transform(X_test)
    test_selected_df = pd.DataFrame(
        X_test_selected,
        columns=chosen_features,
        index=X_test.index,
    )
    test_selected_df["Ucome_fob_ARA"] = y_test.values
    test_selected_df = test_selected_df.reset_index(drop=True)

    return train_selected_df, test_selected_df, list(chosen_features)

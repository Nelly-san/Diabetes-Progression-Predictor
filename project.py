import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    diabetes = load_diabetes()

    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="target")

    print("Dataset shape (rows, columns):", X.shape)
    print("Target length:", len(y))
    print("\nFirst 5 rows of features:\n", X.head())
    print("\nFirst 5 target values:\n", y.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTrain size:", X_train.shape[0], "rows")
    print("Test size:", X_test.shape[0], "rows")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("\n=== Model Evaluation (on TEST set) ===")
    print(f"R²:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    coef_table = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", ascending=False)

    print("\nIntercept (b0):", model.intercept_)
    print("\n=== Coefficients (b1..bk) ===")
    print(coef_table.to_string(index=False))

    sample_row = X_test.iloc[[0]]
    sample_true = y_test.iloc[0]
    sample_pred = model.predict(sample_row)[0]

    print("\n=== Single Prediction Example ===")
    print("Input row (first test sample):")
    print(sample_row.to_string(index=False))
    print("True target:", sample_true)
    print("Predicted target:", round(sample_pred, 2))


if __name__ == "__main__":
    main()
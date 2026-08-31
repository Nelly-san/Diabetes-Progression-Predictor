Diabetes Progression Predictor (Linear Regression)

A small machine learning project that predicts diabetes disease progression using linear regression, built on scikit-learn's built-in diabetes dataset.

What it does

Trains a linear regression model on patient data (age, BMI, blood pressure, and other measurements) to predict a numeric score representing disease progression one year after baseline. Prints out the dataset shape, model accuracy, feature coefficients, and a sample prediction.

Built with
- Python
- pandas
- scikit-learn

How it works
1. Loads the diabetes dataset from `sklearn.datasets` and converts it into a pandas DataFrame for easier inspection.
2. Splits the data into training (80%) and test (20%) sets.
3. Fits a `LinearRegression` model on the training data.
4. Evaluates the model on the held-out test set using three metrics:
   - **R²** — how much of the variance in the outcome the model explains
   - **MAE** (Mean Absolute Error) — average size of prediction errors
   - **RMSE** (Root Mean Squared Error) — similar to MAE but penalizes larger errors more
5. Prints out each feature's coefficient, sorted from most positive to most negative impact on the prediction.
6. Runs one single prediction on a test sample and compares it to the actual value, just to see the model in action on a real example.

Notes

This was a hands-on exercise in the basic machine learning workflow — load data, split it, train a model, evaluate it properly on unseen data, and interpret what the model actually learned (via the coefficients) rather than just treating it as a black box.

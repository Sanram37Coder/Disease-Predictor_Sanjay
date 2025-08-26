# Disease-Predictor_Sanjay
Disease Predictor Workshop Devtown
# Heart Disease Prediction

This notebook demonstrates a machine learning workflow to predict the presence of heart disease using the Heart Disease UCI dataset.

## Workflow:

1.  **Data Loading and Exploration:**
    *   Load the dataset from Kaggle.
    *   Display the first few rows and column names.
    *   Check for missing values and handle them by imputing the mean for numeric columns and a placeholder for categorical columns.

2.  **Data Visualization:**
    *   Generate histograms for numerical features to understand their distributions.
    *   Create a heatmap to visualize correlations between numerical features.

3.  **Data Preprocessing:**
    *   Define features (X) and target variable (y), where the target is a binary indicator of heart disease presence.
    *   Apply one-hot encoding to categorical features.
    *   Split the data into training and testing sets.
    *   Standardize numerical features using `StandardScaler`.

4.  **Model Training and Evaluation:**
    *   Train a Logistic Regression model and evaluate its accuracy and classification report.
    *   Train a Random Forest Classifier and evaluate its accuracy.
    *   Display the confusion matrix for the Logistic Regression model.
    *   Visualize the feature importance from the Random Forest model.

5.  **Model Deployment Preparation:**
    *   Save a sample of the processed data as a CSV template (`Heart_user_template.csv`) for making predictions on new data.
    *   Save the trained Random Forest model and the scaler object using `joblib` for future use.

6.  **Prediction on New Data:**
    *   Load a new dataset (`heart_dataset.csv`) with user data.
    *   Preprocess the new data using the saved scaler and apply one-hot encoding.
    *   Align the columns of the new data with the training data.
    *   Load the saved Random Forest model.
    *   Make predictions on the new data.
    *   Add the predictions as a new column to the user data DataFrame and display the results.

## Dependencies:

The following libraries are used in this notebook:

*   `pandas`
*   `numpy` (often used with pandas)
*   `matplotlib`
*   `seaborn`
*   `sklearn`
*   `joblib`
*   `google.colab` (for file uploads)

## Usage:

1.  Run the cells sequentially to execute the entire workflow.
2.  Upload your Kaggle API key (`Eg:kaggle.json`) when prompted.
3.  Upload the new heart disease data (`Eg:heart_dataset.csv`) when prompted in the "Prediction on New Data" section.

## Data:

The dataset used is the Heart Disease UCI dataset, downloaded from Kaggle.

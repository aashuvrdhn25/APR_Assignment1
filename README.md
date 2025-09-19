Diabetes Prediction using Linear Regression
Overview

This project applies Linear Regression on a diabetes dataset to predict whether a patient is diabetic or not based on medical attributes.
Although Linear Regression is primarily designed for continuous outputs, in this project it is adapted for binary classification (0 = Non-diabetic, 1 = Diabetic) by applying a decision threshold.

Dataset

The dataset (diabetesData.csv) contains several medical predictor variables and one target variable:

Pregnancies – Number of times pregnant

Glucose – Plasma glucose concentration

BloodPressure – Diastolic blood pressure (mm Hg)

SkinThickness – Triceps skin fold thickness (mm)

Insulin – 2-hour serum insulin (mu U/ml)

BMI – Body Mass Index (weight in kg / (height in m)²)

DiabetesPedigreeFunction – Diabetes pedigree function

Age – Age (years)

Outcome – Target variable (0 = Non-diabetic, 1 = Diabetic)

Installation & Setup

Open Google Colab or any Python IDE (with scikit-learn, pandas, matplotlib installed).

Upload the dataset (diabetesData.csv) into your Colab environment.

Copy the provided Python code into a notebook cell.

Run the code to train and evaluate the model.

Required Python libraries:

pip install pandas numpy matplotlib scikit-learn

Results

Outputs continuous predictions that are thresholded at 0.5 to classify patients.

Evaluation metrics used: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² Score.

A scatter plot of Actual vs Predicted values shows how well the model fits.

Limitations

Linear Regression is not ideal for binary classification since it predicts continuous values outside [0,1].

For classification problems like this, Logistic Regression or other machine learning algorithms are more suitable.

Conclusion

This project demonstrates how Linear Regression can be applied to a medical dataset for predictive analysis. While not the best choice for binary outcomes, it provides insights into feature importance and serves as a baseline model.

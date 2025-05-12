# Supervised Learning: Regression Analysis

This project demonstrates various regression techniques, including Least Squares, Gradient Descent for Linear Regression, and Polynomial Regression (implemented with both scikit-learn and manual Gradient Descent). The primary goal is to fit models to a given dataset and visualize the results.

This script was developed as part of a Knowledge Discovery and Data Mining course.

## Project Overview

The Python script (`A. Supervised Learning (SL) - Regression.py`) performs the following key steps:

1.  **Data Acquisition**:
    * Loads data from an Excel file named `Data Take Home Assignment 1 Exercise A.xlsx`.
    * Selects a specific subset of the data for analysis (rows corresponding to group 17, assuming 20 data points per group).

2.  **Data Transformation**:
    * Applies Min-Max Normalization to the features 'X' and 'Y' to scale them into a range of [0, 1]. This helps in stabilizing the gradient descent process.

3.  **Least Squares Linear Regression**:
    * Implements a simple linear regression model of the form $Y = A + BX$.
    * Uses pre-defined (or manually derived) coefficients A and B.
    * Visualizes the original data points and the fitted regression line.

4.  **Gradient Descent for Linear Regression**:
    * Manually implements the gradient descent algorithm to find the optimal parameters (A and B) for the linear model.
    * Iteratively updates the parameters by minimizing the cost function $J = \frac{1}{2} \sum (Y_{actual} - Y_{predicted})^2$.
    * The script shows the state of the regression line after several iterations, demonstrating the convergence process.
    * A learning rate of 0.01 is used.

5.  **Polynomial Regression (Degree 2) using Scikit-learn**:
    * Utilizes `sklearn.preprocessing.PolynomialFeatures` to create polynomial features (degree 2) from the original 'X' feature.
    * Fits a polynomial regression model ($Y = A + BX + CX^2$) using `sklearn.linear_model.LinearRegression`.
    * Visualizes the original data points and the fitted polynomial curve.

6.  **Gradient Descent for Polynomial Regression (Degree 2)**:
    * Manually implements the gradient descent algorithm to find the optimal parameters (A, B, and C) for a quadratic model.
    * Iteratively updates the parameters by minimizing a cost function $J = \frac{1}{4} \sum (Y_{actual} - Y_{predicted})^4$. *Note: This is a custom cost function used for this exercise.*
    * The script shows the state of the regression curve after several iterations.
    * A learning rate of 0.5 is used.

## Requirements

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

You can install the necessary libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn openpyxl

# Housing Price Prediction and Fairness Analysis

## Overview

This UC Berkeley Data 100 project predicts the log sale price of homes using a linear regression model trained on a Cook County dataset of more than 200,000 records. Beyond raw prediction accuracy, the project investigates whether the model's errors fall unevenly across cheap and expensive homes, connecting the analysis to the real history of regressive property tax assessment in Cook County. **It's an end to end supervised regression pipeline** built in Python that predicts housing sale prices on a large Cook County dataset, with a fairness analysis of how the model's errors are distributed across price ranges.

## Features

**Feature Engineering Pipeline**: Encapsulates all transformations in a single reusable function with separate logic for training and test data to prevent data leakage. Handles log transforms, missing value imputation, and one hot encoding of categorical features.

**Model Training and Validation**: Fits a linear regression model using scikit-learn and reaches a training RMSE of 0.61 on the log price scale. Implements k fold cross validation from scratch to estimate test error before submission.

**Custom Loss Optimization**: Trains a linear model under a custom Mean Absolute Percentage Error loss using scipy.optimize, going beyond the standard scikit-learn fit to control exactly what error the model minimizes.

**Fairness Analysis**: Compares RMSE against MAPE across price intervals to reveal a regressive bias where cheap homes are systematically overvalued and expensive homes undervalued, a pattern standard dollar scale RMSE conceals.

## Tech Stack

Python, scikit-learn, scipy, NumPy, Pandas, Matplotlib. Linear regression with ordinary least squares, k fold cross validation, and custom loss optimization.

## Running the Project

Open the notebook in Jupyter and run the cells in order. Make sure scikit-learn, scipy, NumPy, Pandas, and Matplotlib are installed in your environment.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
features = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
features_train = features[:-20]
features_test = features[-20:]

# Split the targets into training/testing sets
target_train = diabetes.target[:-20]
target_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(features_train, target_train)

# Make predictions using the testing set
target_pred = regr.predict(features_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(target_test, target_pred))

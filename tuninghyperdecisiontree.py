import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

mnist = skds.fetch_openml('mnist_784', as_frame=False, parser='auto')

# Extract features (X) and labels (y)
X, y = mnist.data, mnist.target

# Normalize pixel values

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Flatten the images
X.reshape(X.shape[0], -1)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)



#decision tree
dt= DecisionTreeClassifier(max_depth=10, random_state=101)
dt.fit(X_train, y_train)

param_grid ={
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80],
    'min_samples_leaf': [ 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': [ 'auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(dt, param_grid)
grid_search.fit(X_train, y_train)
print("best params", grid_search.best_params_)
best=grid_search.best_estimator_
accuracy=best.score(X_test, y_test)
print("accuracy of best model is", accuracy)

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skds
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
rf= RandomForestClassifier()
rf.fit(X_train, y_train)

param_grid ={
    'n_estimators': [50],
    'max_depth': [10, 30, 50, 70],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [2, 5, 8],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
    'max_features': [ 'auto', 'sqrt']
}
random_search = RandomizedSearchCV(rf, param_grid)
random_search.fit(X_train, y_train)
print("best params", random_search.best_params_)
best=random_search.best_estimator_
accuracy=best.score(X_test, y_test)
print("accuracy of best model is", accuracy)

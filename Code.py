import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as skds
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, roc_curve, precision_score, recall_score, f1_score, roc_auc_score
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
#tuned
dt= DecisionTreeClassifier(max_depth=20, max_features='sqrt',min_samples_leaf=2,min_samples_split=10, random_state=101)
dt.fit(X_train, y_train)


# Random Forest Regressor
rf = RandomForestClassifier( max_depth=50,random_state=101, n_estimators=50, min_samples_leaf=2, min_samples_split= 5, max_features='sqrt', criterion='entropy', bootstrap=False)
rf.fit(X_train, y_train)

# Model Performance
models = [rf,dt]
model_names = [ "Random Forest Classifier", "Decision Tree"]

for model, name in zip(models, model_names):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r_squared = model.score(X_test, y_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print(name)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R-squared:', r_squared)
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(name+'Confusion Matrix')
    plt.show()

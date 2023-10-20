# Pranjal Aggarwal
# 10/october/2023

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
print(iris.keys())
print(iris.target)
x = iris.data
x_train = x[:130]
x_test = x[130:]
y = iris.target
y_train = y[:130]
y_test = y[130:]

# Create and train the KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

# Make predictions on the test data
y_predicted = clf.predict(x_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"Accuracy: {accuracy}")

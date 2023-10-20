# Pranjal Aggarwal
# 11/october/2023


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
diabetes = datasets.load_diabetes()
# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
print(diabetes.data_filename)
# x = diabetes.data[:,np.newaxis,2]
x = diabetes.data
x_train = x[:-30]
x_test = x[-30:]
y_train = diabetes.target[:-30]     #all data except last 30
y_test = diabetes.target[-30:]
# print(x)
model = linear_model.LinearRegression()
model.fit(x_train,y_train)
y_predicted = model.predict(x_test)
# print("mean squared error is ", mean_squared_error(y_test,y_predicted))
# print("coefficient :",model.coef_)
#
# print("bias:",model.intercept_)
# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_predicted)
# plt.show()

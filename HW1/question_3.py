import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from model import logisticRegression

plt.figure(1)
# Load the .mat file
train_data = scipy.io.loadmat('train.mat')
# print (train_data)
# Access a variable from the .mat file
x1 = train_data['x1'].flatten()
x2 = train_data['x2'].flatten()
y = train_data['y'].flatten()

plt.scatter(x1,x2, c=y, label='data point')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('train.mat')
plt.legend()
plt.show()

m = len(y)
x_feature = np.append(np.ones((m,1)),np.array([x1,x2]).T,axis=1)
y = np.array(y).reshape(-1, 1)

logistic_regression = logisticRegression()
theta = np.ones((3,1))
print(theta)
theta, cost = logistic_regression.gradient(x_feature,y,theta,Learningrate=0.05,epochs=15)
print(theta)
y_pred = logistic_regression.sigmod(x_feature.dot(theta))

y_p = (y_pred > 0.5).astype(int).flatten()
# fig , ax = plt.subplots(1,2, sharex=True, sharey=True,)
# ax[0].scatter(x1,x2, c=y_p, label='data point')
# ax[1].scatter(x1,x2, c=y, label='data point')

# x1_boundary = np.linspace(x_feature[:, 1].min(), x_feature[:, 1].max(), 100)
# x2_boundary = -(theta[0] + theta[1] * x1_boundary) / theta[2]
# ax[0].plot(x1_boundary, x2_boundary, color='r', label='Decision Boundary')
# ax[0].set_title('predict')
# ax[1].set_title('ans')
# plt.suptitle('train.mat')
# plt.legend(loc='upper right')
# plt.show()
fig , ax = plt.subplots(1,2, sharex=True, sharey=True)
test_data = scipy.io.loadmat('test.mat')
# print (test_data)
test_x1 = test_data['x1'].flatten()
test_x2 = test_data['x2'].flatten()
test_y = test_data['y'].flatten()
m = len(test_y)
x_test = np.append(np.ones((m,1)),np.array([test_x1,test_x2]).T,axis=1)
y_pred = logistic_regression.sigmod(x_test.dot(theta))

y_p = (y_pred > 0.5).astype(int).flatten()
print(y_p)
print(test_y)
misclassified_samples = np.sum(y_p != test_y)
print(misclassified_samples)
error_rate = (misclassified_samples / len(test_y)) * 100
print("Test Error: {:.2f}%".format(error_rate))

ax[0].scatter(test_x1,test_x2, c=y_p, label='data point')
ax[1].scatter(test_x1,test_x2, c=test_y, label='data point')

x1_boundary = np.linspace(x_test[:, 1].min(), x_test[:, 1].max(), 100)
x2_boundary = -(theta[0] + theta[1] * x1_boundary) / theta[2]
# x2_boundary = -(theta[0]  * x1_boundary) / theta[1]
ax[0].plot(x1_boundary, x2_boundary, color='r', label='Decision Boundary')
ax[0].set_title('predict')
ax[1].set_title('ans')
ax[0].set_xlim(min(test_x1)-0.2,max(test_x1)+0.2)
ax[0].set_ylim(min(test_x2)-0.2,max(test_x2)+0.2)
plt.suptitle('test.mat')
plt.legend(loc='upper right')
plt.show()
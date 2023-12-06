import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from model import linear_Regression
# 1.1
# Load the .mat file
mat_data = scipy.io.loadmat('data.mat')

# Access a variable from the .mat file
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()

data_df = pd.DataFrame({'x':x, 'y':y})
print(data_df.info())
plt.scatter(x,y, label='data point',c ='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('1.1')
plt.legend()
plt.show()
#1.2
linear_regression = linear_Regression()
theta = np.zeros((2,1))

m = len(x)
x_feature = np.append(np.ones((m,1)),np.array(x).reshape(-1,1),axis=1)
y = np.array(y).reshape(-1, 1)
theta, cost = linear_regression.gradient_descent(x_feature,y,theta,Learningrate=0.1,epochs=1000)
print("h(x) = {} + {}x".format(str(round(theta[0, 0], 2)),
                               str(round(theta[1, 0], 2))))
print(theta)
print(cost[0],cost[999])
x_value = [x / 100 for x in range(0, 100)]
y_value=[(x * theta[1] + theta[0]) for x in x_value]
plt.plot(x_value, y_value, color='red', linewidth=2, label='預測線')
plt.scatter(x,y, label='data point',color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('1.2')
plt.show()

# 1.3
theta = np.zeros((3,1))
m = len(x)
x_feature = np.append(np.ones((m,1)),np.array([x,x**2]).T,axis=1)
y = np.array(y).reshape(-1, 1)
theta, cost = linear_regression.gradient_descent(x_feature,y,theta,Learningrate=0.3,epochs=1000)
print("h(x) = {} + {}x + {}x^2".format(str(round(theta[0, 0], 2)),
                                       str(round(theta[1, 0], 2)),
                                       str(round(theta[2, 0], 2))))
print(theta)
print(cost[0],cost[999])
x_value = [x / 100 for x in range(0, 100)]
y_value=[((x**2)*theta[2] + x * theta[1] + theta[0]) for x in x_value]
plt.plot(x_value, y_value, color='red', linewidth=2, label='預測線')
plt.scatter(x,y, label='data point',color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('1.3')
plt.show()
# 1.4
theta = np.zeros((5,1))
m = len(x)
x_feature = np.append(np.ones((m,1)),np.array([x,x**2,x**3,x**4]).T,axis=1)
y = np.array(y).reshape(-1, 1)
theta, cost = linear_regression.gradient_descent(x_feature,y,theta,Learningrate=0.3,epochs=1000)
print("h(x) = {} + {}x + {}x^2 + {}x^3 + {}x^4".format(str(round(theta[0, 0], 2)),
                                                       str(round(theta[1, 0], 2)),
                                                       str(round(theta[2, 0], 2)),
                                                       str(round(theta[3, 0], 2)),
                                                       str(round(theta[4, 0], 2))))
print(theta)
print(cost[0],cost[999])
x_value = [x / 100 for x in range(0, 100)]
y_value=[((x**4)*theta[4] + (x**3)*theta[3] + (x**2)*theta[2] + x * theta[1] + theta[0]) for x in x_value]
plt.plot(x_value, y_value, color='red', linewidth=2, label='預測線')
plt.scatter(x,y, label='data point',color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('1.4')
plt.show()
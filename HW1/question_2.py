import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from model import linear_Regression

# Load the .mat file
mat_data = scipy.io.loadmat('data.mat')
# Access a variable from the .mat file
x = mat_data['x'].flatten()
y = mat_data['y'].flatten()

num_samples = 30
num_iterations = 200

selected_data = []
# plt.figure(0)
for _ in range(num_iterations):
    # Randomly select 30 indices without replacement
    indices = np.random.choice(len(x), num_samples, replace=False)
    selected_x = x[indices]
    selected_y = y[indices]
    # plt.scatter(selected_x ,selected_y , label='data point',color='b')
    selected_data.append((selected_x, selected_y))

    # plt.show()

# for _ in range(num_iterations):
#     # Randomly select a starting index
#     start_index = np.random.randint(0, len(x) - num_samples)
#     end_index = start_index + num_samples
#     selected_x = x[start_index:end_index]
#     selected_y = y[start_index:end_index]

#     # Plot the selected data (optional)
#     plt.scatter(selected_x, selected_y, label='data point', color='b')

#     selected_data.append((selected_x, selected_y))

# # # Show the plot (optional)
#     plt.show()

print(len(selected_data))

linear_regression = linear_Regression()
theta = np.zeros((2,1))
plt.figure(1)
model_fits =[]
bias_list =[]
for i,data in enumerate(selected_data):
    # print(x)
    x_feature = np.append(np.ones((num_samples,1)),np.array(data[0]).reshape(-1,1),axis=1)
    y = np.array(data[1]).reshape(-1, 1)
    theta, cost = linear_regression.gradient_descent(x_feature,y,theta,Learningrate=0.1,epochs=1000)
    # print("h(x) = {} + {}x".format(str(round(theta[0, 0], 2)),
    #                                str(round(theta[1, 0], 2))))
    # print(cost[0],cost[999])
    y_pred = np.dot(x_feature,theta)
    x_value = [x / 100 for x in range(0, 100)]
    y_value=[(x * theta[1] + theta[0]) for x in x_value]
    model_fits.append(y_value)
    bias_list.append(np.abs(y_pred-y))
    plt.plot(x_value, y_value, linewidth=2, label='預測線')
mean_fit = np.mean(model_fits,axis = 0)
variance = np.mean(np.mean((model_fits - mean_fit) **2,axis=0))
bias =np.mean(bias_list)
print("line")
print("variance: ",variance)
print("bias :",bias)
# print("mean ", mean_fit)
# a = (model_fits - mean_fit) **2

# print(len(mean_fit))
# print(len(variance))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('lines')
plt.show()


theta = np.zeros((5,1))
plt.figure(2)
model_fits =[]
bias_list =[]
for i,data in enumerate(selected_data):
    x = data[0]
    x_feature = np.append(np.ones((num_samples,1)),np.array([x,x**2,x**3,x**4]).T,axis=1)
    y = np.array(data[1]).reshape(-1, 1)
    theta, cost = linear_regression.gradient_descent(x_feature,y,theta,Learningrate=0.3,epochs=1000)
    # print("h(x) = {} + {}x + {}x^2 + {}x^3 + {}x^4".format(str(round(theta[0, 0], 2)),
    #                                                     str(round(theta[1, 0], 2)),
    #                                                     str(round(theta[2, 0], 2)),
    #                                                     str(round(theta[3, 0], 2)),
    #                                                     str(round(theta[4, 0], 2))))
    # print(theta)
    # print(cost[0],cost[999])
    y_pred = np.dot(x_feature,theta)
    x_value = [x / 100 for x in range(0, 100)]
    y_value=[((x**4)*theta[4] + (x**3)*theta[3] + (x**2)*theta[2] + x * theta[1] + theta[0]) for x in x_value]
    plt.plot(x_value, y_value, linewidth=2, label='預測線')
    model_fits.append(y_value)
    bias_list.append(np.abs(y_pred-y))

mean_fit = np.mean(model_fits,axis = 0)
variance = np.mean(np.mean((model_fits - mean_fit) **2,axis=0))
bias = np.mean(bias_list)
print("quartic curves")
print("variance: ",variance)
print("bias: ",bias)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('quartic curves')
plt.show()
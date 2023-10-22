from __future__ import print_function
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import cv2 

# input image dimensions 28x28 
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

count = [0 for _ in range(10)]
print(count)
my_x_dataset = []
my_y_dataset = []
for i, data in enumerate(x_train):
    if sum(count) == 5000:
        break
    if count[y_train[i]] == 500:
        continue
    my_x_dataset.append(data)
    my_y_dataset.append(y_train[i])
    count[y_train[i]] +=1
print(count)
# print(my_y_dataset)
my_x_dataset = np.array(my_x_dataset)
my_y_dataset = np.array(my_y_dataset)
# 4.1
amount= 50
lines = 5
columns = 10
number = np.zeros(amount)

for i in range(amount): 
    number[i] = my_y_dataset[i] 
    # print(number[i])

fig = plt.figure()

for i in range(amount):
    ax = fig.add_subplot(lines, columns, 1 + i) 
    plt.imshow(my_x_dataset[i,:,:], cmap='gray') 
    plt.sca(ax)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
plt.show()

# 4.2 
# np.set_printoptions(threshold=np.inf)
my_x_dataset = my_x_dataset.reshape(my_x_dataset.shape[0],-1)

mean = np.mean(my_x_dataset)
std = np.std(my_x_dataset)
my_x_dataset_normalized = ((my_x_dataset - mean) / std )
print(mean)
print(std)
print(len(my_x_dataset_normalized))
cov_matrix = np.cov(my_x_dataset_normalized, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

np.set_printoptions(threshold=np.inf)
print("Eigenvalues (sorted):")
print(eigenvalues.shape)

print("\nEigenvectors (sorted by eigenvalues):")
print(eigenvectors)
# print(sorted_indices)

# 4.3
# do pca 
dimensions = [500, 300, 100, 50]

for k in dimensions:
    selected_eigenvectors = eigenvectors[:, :k]
    my_x_dataset_pca = my_x_dataset_normalized.dot(selected_eigenvectors)
    x_reconstructed = my_x_dataset_pca.dot(selected_eigenvectors.T)
    for digit in range(10):
        digit_indices = np.where(my_y_dataset == digit)[0]
        for i in range(10):
            # plt.subplot(10, 10, i + 1)
            # plt.imshow(x_train[digit_indices[i]].reshape(28, 28), cmap='gray')
            # plt.axis('off')
            plt.subplot(10, 10, i + 1+(digit*10))
            plt.imshow(x_reconstructed[digit_indices[i]].reshape(28, 28), cmap='gray')
            plt.axis('off')
    plt.suptitle(f' {k} Dimensions')
    plt.show()
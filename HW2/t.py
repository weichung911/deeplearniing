import tensorflow as tf
from tensorflow.keras.applications import AlexNet, ResNet50
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define and compile AlexNet model
alexnet_model = AlexNet(input_shape=(32, 32, 3), include_top=True, weights=None, classes=10)
alexnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train AlexNet model
history_alexnet = alexnet_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate AlexNet model
y_pred_alexnet = alexnet_model.predict(x_test)
accuracy_alexnet = accuracy_score(y_test, y_pred_alexnet.argmax(axis=1))
print(f'AlexNet Accuracy: {accuracy_alexnet}')

# Define and compile ResNet model
resnet_model = ResNet50(input_shape=(32, 32, 3), include_top=True, weights=None, classes=10)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train ResNet model
history_resnet = resnet_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate ResNet model
y_pred_resnet = resnet_model.predict(x_test)
accuracy_resnet = accuracy_score(y_test, y_pred_resnet.argmax(axis=1))
print(f'ResNet Accuracy: {accuracy_resnet}')

# Plot training loss changes
plt.plot(history_alexnet.history['loss'], label='AlexNet Training Loss')
plt.plot(history_resnet.history['loss'], label='ResNet Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

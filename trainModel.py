import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalise data to range of [0, 1]
x_train = x_train / 255
x_test = x_test / 255

print(x_train.shape)

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

model.save('mnist_model.h5')
print("Model saved as mnist_model.h5")



# Testing out


# # Pick a random test image
# random_index = np.random.randint(0, len(x_test))
# test_image = x_test[random_index]

# # Save the test image to a file
# plt.imshow(test_image, cmap='gray')
# plt.savefig('test_image.png')

# print('Test image saved as "test_image.png"')

# # Expand dimensions since the model expects a batch (even if it's just one sample)
# test_image = np.expand_dims(test_image, axis=0)

# # Predict the digit using the model
# predictions = model.predict(test_image)

# # Get the index of the highest probability
# predicted_digit = np.argmax(predictions[0])

# print(f'Predicted digit: {predicted_digit}')
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

model = tf.keras.models.load_model('mnist_model.h5')

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255.0

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

random_index = random.randint(0, len(x_test) - 1)
test_image = x_test[random_index]

plt.imshow(test_image, cmap='gray')
plt.savefig('test_image.png')
print("Test image saved as 'test_image.png'")

test_image = np.expand_dims(test_image, axis=0)

predictions = model.predict(test_image)
predicted_digit = np.argmax(predictions[0])
print(f'Predicted digit: {predicted_digit}')
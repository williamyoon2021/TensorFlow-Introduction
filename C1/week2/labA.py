import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.80):
            self.model.stop_training = True

callbacks = myCallback()
# Load Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist

# Load training and test sets
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# EXAMPLE CODE - Not involved with actual model
index = 42
np.set_printoptions(linewidth=320)
print(f'LABEL: {training_labels[index]}')
# Visualize index 0
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')
plt.imshow(training_images[index])
# END OF EXAMPLE CODE

# Normalize pixel values to 1
training_images = training_images / 255.0
test_images = test_images / 255.0

# Classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# EXAMPLE CODE - Not involved with actual model
# Declare sample inputs and convert to tensor
inputs = np.array([[1.0, 44.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Inputs into softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output to softmax function: {outputs.numpy()}')

# Sum of all values after softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Index with highest value (index 1 since 44 is largest)
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')
# END OF EXAMPLE CODE

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

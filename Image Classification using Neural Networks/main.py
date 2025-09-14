import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10, 
#           validation_data=(test_images, test_labels))

# loss, accuracy = model.evaluate(test_images, test_labels)
# print(f"Test accuracy: {accuracy * 100:.2f}%")
# print(f"Test loss: {loss:.2f}")

# # Save the trained model
# model.save('cifar10_model.h5')
# models.load_model('cifar10_model.h5')

model = models.load_model('cifar10_model.h5')

model


# Load the images
img_pil = tf.keras.utils.load_img("download.jpg", target_size=(32, 32))
img_array = tf.keras.utils.img_to_array(img_pil)
img_batch = np.expand_dims(img_array, axis=0)

plt.imshow(img_pil)
prediction = model.predict(img_batch / 255.0)
index = np.argmax(prediction)
print("The image most likely belongs to {}".format(class_names[index]))
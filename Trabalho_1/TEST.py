import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the images
images = []
labels = []
for i, animal_dir in enumerate(os.listdir('C:/Users/conta/Desktop/animals10/raw-img')):
    for filename in os.listdir(os.path.join('C:/Users/conta/Desktop/animals10/raw-img', animal_dir)):
        try:
            img = Image.open(os.path.join('C:/Users/conta/Desktop/animals10/raw-img', animal_dir, filename))
            img = img.resize((128, 128))
            img_arr = np.array(img)
            if img_arr.shape != (128, 128, 3):
                raise ValueError('Image has unexpected shape: %s' % str(img_arr.shape))
            images.append(img_arr)
            labels.append(i)
        except Exception as e:
            print('Error loading image %s: %s' % (os.path.join('C:/Users/conta/Desktop/animals10/raw-img', animal_dir, filename), str(e)))
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

# Preprocess the data
train_labels = keras.utils.to_categorical(train_labels, num_classes=2)
test_labels = keras.utils.to_categorical(test_labels, num_classes=2)

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Create the data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Create the model
model = keras.Sequential([
    layers.Conv2D(50, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=500),
                    steps_per_epoch=len(train_images) // 500, epochs=500,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Visualize the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(val_acc) - 0.1, 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(val_loss)])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


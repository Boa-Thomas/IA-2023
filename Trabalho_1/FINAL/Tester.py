import tensorflow as tf
from tensorflow import keras
from PIL import Image
# Definir lista com as classes
classes = [ "VACA","OVELHA"]
import numpy as np

# Load the saved model
model = keras.models.load_model('Model-500.h5')

# Load the image
img = Image.open('cow.jpg')

# Preprocess the image
img = img.resize((128, 128))  # Resize the image to the input size of your model
img = keras.preprocessing.image.img_to_array(img)  # Convert the image to a numpy array
img = img / 255.0  # Normalize the pixel values to [0, 1]
img = tf.expand_dims(img, 0)  # Add a batch dimension

# Use the model for prediction
predictions = model.predict(img)

# Print the predictions
print(predictions)

# Get the index of the highest prediction value
predicted_index = np.argmax(predictions[0])

# Get the corresponding class name
predicted_class = classes[predicted_index]

# Print the classification result
print(f"Predicted class: {predicted_class}")
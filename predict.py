import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("model.h5")

# Image path (FIX THIS LINE)
img_path = "test_image.jpg/test..jpg"

# Load image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
class_index = np.argmax(prediction)

class_names = ['Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight']

print("Predicted class name:", class_names[class_index])
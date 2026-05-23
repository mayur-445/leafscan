from flask import Flask, request, render_template
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

# Initialize app (IMPORTANT FIX)
app = Flask(__name__)

# Load model
from keras.models import load_model
model = tf.keras.models.load_model("model.keras")  # Load without compiling for faster startup

# Class names
class_names = ['Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight']

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        result = f"{class_names[class_index]} ({confidence*100:.2f}%)"

    return render_template('index.html', result=result)


# 🔥 IMPORTANT PART (FIXED)
if __name__ == "__main__":
    print("SERVER STARTING...")
    app.run(host="0.0.0.0", port=5000, debug=True)
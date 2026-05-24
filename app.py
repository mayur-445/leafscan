from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize app (IMPORTANT FIX)
app = Flask(__name__)

# Load model
from keras.models import load_model
model = load_model("model.h5", compile=False)  # Load without compiling for faster startup

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
        print("saved file path:", filepath)

        # Process image
        from PIL import Image
        img = Image.open(filepath)
        img = img.resize((128, 128))

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = class_names[class_index]
        confidence = round(np.max(prediction) * 100, 2)

        return render_template('index.html', prediction=result, confidence=confidence,image_path=filepath)
        return render_template('index.html')


# 🔥 IMPORTANT PART (FIXED)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
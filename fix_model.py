import tensorflow as tf
from tensorflow.keras.models import load_model

# Load old model
model = load_model("model.h5", compile=False,safe_mode=False)

# Save again in compatible format
model.save("model.keras")

print("Model converted successfully!")
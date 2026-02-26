from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained MLP model
model = load_model("mnist_cnn_model.h5")   # or mlp.h5 if named differently

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]

    # Decode base64 image
    image_data = base64.b64decode(data.split(",")[1])
    image = Image.open(io.BytesIO(image_data)).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy
    image = np.array(image)

    # Normalize (0–255 → 0–1)
    image = image / 255.0

    # Reshape for CNN (IMPORTANT FIX ✅)
    image = image.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(image)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "prediction": predicted_digit,
        "confidence": round(confidence, 2)
    })



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

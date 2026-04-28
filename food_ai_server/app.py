from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model(
    "keras_model.h5",
    compile=False
)

# Đọc labels đúng thứ tự
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = []
    for line in f.readlines():
        text = line.strip()

        # bỏ số đầu dòng
        if len(text) > 2:
            labels.append(text[2:].strip())
        else:
            labels.append(text.strip())

# Bảng calo
calories = {
    "Phở": 450,
    "Cơm Trắng": 300,
    "Gà Rán": 700,
    "Bánh Mì": 380,
    "Bánh Cuốn": 350,
    "Gỏi Cuốn": 50
}

@app.route('/predict', methods=['POST'])
def predict():

    # Kiểm tra ảnh
    if 'image' not in request.files:
        return jsonify({
            "food": "Không nhận được ảnh",
            "calo": 0,
            "confidence": 0
        })

    file = request.files['image']

    # Xử lý ảnh
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.asarray(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    # Làm sạch tên món
    food = labels[index].strip()

    # Debug terminal
    print("Prediction:", prediction)
    print("Index:", index)
    print("Food =", repr(food))
    print("Confidence:", confidence)

    return jsonify({
        "food": food,
        "calo": calories.get(food, 0),
        "confidence": round(confidence * 100, 2)
    })

app.run(host="0.0.0.0", port=5000)
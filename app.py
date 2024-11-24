from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'waste_classifier_model.h5')

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['Kaca', 'Kardus', 'Kertas', 'Logam', 'Plastik', 'Residu']
recommendations = {
    'Kaca': [
        {'title': 'Reusable Bottles', 'link': 'https://example.com/reusable-bottles'},
        {'title': 'Eco-Friendly Bags', 'link': 'https://example.com/eco-friendly-bags'},
        {'title': 'Recycled Plastic Items', 'link': 'https://example.com/recycled-plastic-items'}
    ],
    'Kardus': [
        {'title': 'Recycled Notebooks', 'link': 'https://example.com/recycled-notebooks'},
        {'title': 'Compostable Paper', 'link': 'https://example.com/compostable-paper'},
        {'title': 'Paper Crafts', 'link': 'https://example.com/paper-crafts'}
    ],
    'Kertas': [
        {'title': 'Glass Containers', 'link': 'https://example.com/glass-containers'},
        {'title': 'Recycled Glassware', 'link': 'https://example.com/recycled-glassware'},
        {'title': 'Artistic Glass Products', 'link': 'https://example.com/artistic-glass-products'}
    ],
    'Logam': [
        {'title': 'Reusable Utensils', 'link': 'https://example.com/reusable-utensils'},
        {'title': 'Recycled Metal Art', 'link': 'https://example.com/recycled-metal-art'},
        {'title': 'Metal Containers', 'link': 'https://example.com/metal-containers'}
    ],
    'Plastik': [
        {'title': 'Reusable Utensils', 'link': 'https://example.com/reusable-utensils'},
        {'title': 'Recycled Metal Art', 'link': 'https://example.com/recycled-metal-art'},
        {'title': 'Metal Containers', 'link': 'https://example.com/metal-containers'}
    ],
    'Residu': [
        {'title': 'Reusable Utensils', 'link': 'https://example.com/reusable-utensils'},
        {'title': 'Recycled Metal Art', 'link': 'https://example.com/recycled-metal-art'},
        {'title': 'Metal Containers', 'link': 'https://example.com/metal-containers'}
    ]
}


IMG_SIZE = (224, 224)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file:
        try:
            img = preprocess_image(file.read())

            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            reusable_goods = recommendations.get(predicted_class, [])

            return jsonify({
                'class': predicted_class,
                'confidence': float(confidence),
                'recommendations': reusable_goods
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file'}), 400


if __name__ == '__main__':
    app.run()

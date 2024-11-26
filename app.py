from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('model', 'best_model_classification.keras')

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
        {'title': ' Crafting Miniature Masterpieces: Upcycling Aluminum Cans into Tiny Pots', 'link': 'https://youtube.com/shorts/00L3nudLdGo?si=1p7SY4gemVjfZETS'},
        {'title': 'Recycled Metal Art', 'link': 'https://example.com/recycled-metal-art'},
        {'title': 'Metal Containers', 'link': 'https://example.com/metal-containers'}
    ],
    'Plastik': [
        {'title': 'Plastic Bottle Flower Vase DIY Ideas | Home Decor |', 'link': 'https://youtu.be/hYDkLNW4deU?si=8ooHLOd4hsi7VS9z'},
        {'title': 'Upcycled coffee sachet / Plastic wrapper purse', 'link': 'https://youtu.be/uIFvr71I8Go?si=oiciu1Zm0_cw4_vs'},
        {'title': 'DIY Plastic Bottle Enchanted Rose | Best Out of Waste |', 'link': 'https://youtu.be/cGJdhgCA9JE?si=2TL2Y-NFb8HIXJ7U'}
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

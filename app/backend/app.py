# backend/app.py
import os
import numpy as np
import tensorflow as tf
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils import preprocess_image, preprocess_efficientnet_image
from typing import Dict, Any

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
def load_models():
    models = {
        'disease': tf.keras.models.load_model('models/disease_model.h5'),
        'variety': tf.keras.models.load_model('models/variety_model.h5'),
        # 'age_efficientnet': torch.load('models/age_efficientnet.pth')
    }
    return models

MODELS = load_models()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict_all', methods=['POST'])
def predict_all():
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        diseases = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
        varieties = ['ADT45', 'AndraPonni', 'AtchayaPonni', 'IR20', 'KarnatakaPonni', 'Onthanel', 'Ponni', 'RR', 'Surya', 'Zonal']

        # Prediction results
        predictions: Dict[str, Any] = {}
        
        # Variety Prediction
        variety_img = preprocess_image(filepath)
        variety_pred = MODELS['variety'].predict(variety_img)
        predictions['variety'] = {
            'predicted': varieties[np.argmax(variety_pred)],
            'confidence': float(np.max(variety_pred))
        }
        
        # Disease Prediction
        disease_img = preprocess_image(filepath)
        disease_pred = MODELS['disease'].predict(disease_img)
        predictions['disease'] = {
            'predicted': diseases[np.argmax(disease_pred)],
            'confidence': float(np.max(disease_pred))
        }
        
        # Age Prediction (EfficientNet with .pth)
        # try:
        #     # Preprocess for EfficientNet requires both image and variety
        #     # Example preprocessing - you'll need to adjust based on your specific model
        #     variety_index = np.argmax(variety_pred)
        #     age_efficientnet_img = preprocess_efficientnet_image(filepath, variety_index)
            
        #     # Switch model to evaluation mode
        #     MODELS['age_efficientnet'].eval()
            
        #     # Disable gradient computation
        #     with torch.no_grad():
        #         age_efficientnet_pred = MODELS['age_efficientnet'](age_efficientnet_img)
            
        #     predictions['age_efficientnet'] = {
        #         'predicted': float(age_efficientnet_pred.numpy()[0]),
        #         'confidence': 1.0  # Adjust as needed
        #     }
        # except Exception as age_error:
        #     predictions['age_efficientnet'] = {
        #         'error': f'Age prediction (EfficientNet) failed: {str(age_error)}'
        #     }
        
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)
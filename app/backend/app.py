import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from typing import Dict, Any
import tensorflow as tf

# Flask config
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === CLASS DEFINITIONS ===
class AgeRegressor(nn.Module):
    def __init__(self, num_labels, num_varieties):
        super().__init__()
        self.base = models.efficientnet_b0(weights=None)
        self.base.classifier = nn.Identity()
        self.metadata_dim = num_labels + num_varieties + 1
        self.head = nn.Sequential(
            nn.Linear(1280 + self.metadata_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x, label_vec, variety_vec, grvi_scalar):
        features = self.base(x)
        meta = torch.cat([label_vec, variety_vec, grvi_scalar], dim=1)
        combined = torch.cat([features, meta], dim=1)
        return self.head(combined)

# === HELPER FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img

def preprocess_efficientnet_image(filepath):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(filepath).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor

# === LOAD MODELS ===
def load_models():
    models_dict = {
        'disease': tf.keras.models.load_model('../models/diseased_model.h5'),
        'variety': tf.keras.models.load_model('../models/variety_model.h5'),
        'age_efficientnet': AgeRegressor(num_labels=3, num_varieties=17)
    }
    models_dict['age_efficientnet'].load_state_dict(torch.load('../models/age_model.pth', map_location='cpu'))
    models_dict['age_efficientnet'].eval()
    return models_dict

MODELS = load_models()

# === CLASSES ===
diseases = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
varieties = ['ADT45', 'AndraPonni', 'AtchayaPonni', 'IR20', 'KarnatakaPonni', 'Onthanel', 'Ponni', 'RR', 'Surya', 'Zonal']

# === MAIN PREDICTION ENDPOINT ===
@app.route('/predict_all', methods=['POST'])
def predict_all():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        predictions: Dict[str, Any] = {}

        # === 1. Disease prediction ===
        img_np = preprocess_image(filepath)
        disease_pred = MODELS['disease'].predict(img_np)
        label_idx = int(np.argmax(disease_pred))
        label_conf = float(np.max(disease_pred))
        predictions['disease'] = {
            'predicted': diseases[label_idx],
            'confidence': label_conf
        }

        # === 2. Variety prediction ===
        variety_pred = MODELS['variety'].predict(img_np)
        variety_idx = int(np.argmax(variety_pred))
        variety_conf = float(np.max(variety_pred))
        predictions['variety'] = {
            'predicted': varieties[variety_idx],
            'confidence': variety_conf
        }

        # === 3. Age prediction ===
        img_tensor = preprocess_efficientnet_image(filepath)
        label_onehot = torch.zeros((1, 3))
        label_onehot[0, label_idx % 3] = 1.0
        variety_onehot = torch.zeros((1, 17))
        variety_onehot[0, variety_idx] = 1.0
        grvi_scalar = torch.tensor([[0.5]])

        with torch.no_grad():
            age_output = MODELS['age_efficientnet'](img_tensor, label_onehot, variety_onehot, grvi_scalar)
            age_value = int(round(age_output.item()))
        
        predictions['age'] = {
            'predicted': age_value,
            'confidence': 1.0
        }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# === RUN FLASK ===
if __name__ == '__main__':
    app.run(debug=True)

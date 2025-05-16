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
import gdown
import zipfile
import pathlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask config
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Google Drive model URL - extract the file ID for gdown
GDRIVE_FILE_ID = '1eRq0Nqz_pnSLD_WclAWQwuPusd8dJtPm'
MODEL_ZIP_PATH = os.path.join(MODELS_FOLDER, 'models.zip')

# Initialize models as None - will be loaded during app initialization
MODELS = None


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


# === DOWNLOAD MODELS FROM GOOGLE DRIVE ===
def download_models_from_gdrive():
    """Download model files from Google Drive if they don't exist locally"""
    logger.info("Checking for model files...")
    
    # Define the expected model file paths
    disease_model_path = os.path.join(MODELS_FOLDER, 'diseased_model.h5')
    variety_model_path = os.path.join(MODELS_FOLDER, 'variety_model.h5')
    age_model_path = os.path.join(MODELS_FOLDER, 'age_model.pth')
    
    # Check if models already exist
    models_exist = (
        os.path.exists(disease_model_path) and
        os.path.exists(variety_model_path) and
        os.path.exists(age_model_path)
    )
    
    if models_exist:
        logger.info("Model files already exist. Skipping download.")
        return True
    
    try:
        logger.info(f"Downloading models from Google Drive using file ID: {GDRIVE_FILE_ID}")
        # Use the direct file ID format for gdown
        output = gdown.download(id=GDRIVE_FILE_ID, output=MODEL_ZIP_PATH, quiet=False)
        
        if output is None:
            logger.error("Download failed - gdown returned None")
            return False
            
        if not os.path.exists(MODEL_ZIP_PATH):
            logger.error(f"Download failed - zip file not found at {MODEL_ZIP_PATH}")
            return False
            
        logger.info(f"Extracting models to {MODELS_FOLDER}...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        # Clean up zip file after extraction
        os.remove(MODEL_ZIP_PATH)
        logger.info("Models downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading or extracting models: {e}")
        return False


# === LOAD MODELS ===
def load_models():
    global MODELS
    
    # First ensure models are downloaded
    if not download_models_from_gdrive():
        logger.error("Failed to download models. Cannot continue.")
        return False
    
    try:
        logger.info("Loading models into memory...")
        MODELS = {
            'disease': tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'diseased_model.h5')),
            'variety': tf.keras.models.load_model(os.path.join(MODELS_FOLDER, 'variety_model.h5')),
            'age_efficientnet': AgeRegressor(num_labels=3, num_varieties=17)
        }
        MODELS['age_efficientnet'].load_state_dict(torch.load(os.path.join(MODELS_FOLDER, 'age_model.pth'), map_location='cpu'))
        MODELS['age_efficientnet'].eval()
        logger.info("Models loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# Initialize models as None
MODELS = None

# === CLASSES ===
diseases = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
varieties = ['ADT45', 'AndraPonni', 'AtchayaPonni', 'IR20', 'KarnatakaPonni', 'Onthanel', 'Ponni', 'RR', 'Surya', 'Zonal']

# Function to initialize models
def initialize_models():
    """Check and load models if they're not already loaded."""
    global MODELS
    if MODELS is None:
        logger.info("Checking and loading models...")
        if load_models():
            logger.info("Models loaded successfully")
        else:
            logger.warning("Failed to load models")

# Initialize models when app starts (compatible with Flask 2.3+)
with app.app_context():
    initialize_models()

# === MAIN PREDICTION ENDPOINT ===
@app.route('/predict_all', methods=['POST'])
def predict_all():
    global MODELS
    
    # Check if models are loaded
    if MODELS is None:
        initialize_models()
        
    # If models still couldn't be loaded after trying to initialize
    if MODELS is None:
        return jsonify({'error': 'Models could not be loaded. Please check the logs.'}), 500
    
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
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# === Health check endpoint with more details ===
@app.route('/health', methods=['GET'])
def health_check():
    global MODELS
    
    model_status = {}
    if MODELS is not None:
        for model_name in MODELS:
            model_status[model_name] = "loaded"
    
    # Check if model files exist even if not loaded in memory
    disease_model_path = os.path.join(MODELS_FOLDER, 'diseased_model.h5')
    variety_model_path = os.path.join(MODELS_FOLDER, 'variety_model.h5')
    age_model_path = os.path.join(MODELS_FOLDER, 'age_model.pth')
    
    files_exist = {
        'diseased_model.h5': os.path.exists(disease_model_path),
        'variety_model.h5': os.path.exists(variety_model_path),
        'age_model.pth': os.path.exists(age_model_path)
    }
    
    return jsonify({
        'status': 'ok', 
        'models_loaded': MODELS is not None,
        'model_status': model_status,
        'model_files': files_exist
    })

# === Manual model loading endpoint ===
@app.route('/load_models', methods=['POST'])
def load_models_endpoint():
    initialize_models()
    if MODELS is not None:
        return jsonify({'status': 'success', 'message': 'Models loaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load models'}), 500

# === RUN FLASK ===
if __name__ == '__main__':
    # Try to load models at startup when run directly
    if load_models():
        logger.info("Models loaded successfully at startup")
    else:
        logger.warning("Models could not be loaded at startup. They will be loaded on first request.")
    
    port = int(os.environ.get("PORT", 4000))  # Use env PORT or default to 4000
    app.run(debug=True, host='0.0.0.0', port=port)
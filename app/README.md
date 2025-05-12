# Plant AI Prediction App

## Project Overview
This web application allows users to upload plant images and get predictions for plant age, category, and variety using pre-trained AI models.

## Prerequisites
- Python 3.8+
- Node.js 14+
- pip
- npm

## Backend Setup
1. Navigate to the backend directory
```bash
cd backend
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your trained models
- Place your TensorFlow models in the `models/` directory:
  - `age_efficientnet.pth (best_model.pth)`
  - `disease_model.h5`
  - `variety_model.h5`

## Frontend Setup
1. Navigate to the frontend directory
```bash
cd frontend
```

2. Install dependencies
```bash
npm install
```

## Running the Application
1. Start the Backend (in backend/ directory)
```bash
flask run
```

2. Start the Frontend (in frontend/ directory)
```bash
npm start
```

## Customization
- Modify `App.py` to adjust model loading and prediction logic
- Update `utils.py` for specific image preprocessing needs
- Adjust frontend components in React as required

## Troubleshooting
- Ensure model paths are correct
- Check that model input shapes match preprocessing
- Verify dependencies are installed correctly
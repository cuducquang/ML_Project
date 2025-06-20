# ML_Project

**Link to Google Drive (includes datasets and models):**  
[Google Drive Folder](https://drive.google.com/drive/folders/1NVELOEizQM09ootxMvoahPx3gxvsujLZ?usp=drive_link)

**Best models for each task:**
- `best_diseased_model.h5`: [Download](https://drive.google.com/file/d/1VLPAeiCB1CnTR_SRKW7ssXhBXgqoxuUe/view?usp=drive_link)
- `best_variety_model.h5`: [Download](https://drive.google.com/file/d/1FfyDsHhZY70FpgZbuP7gve71fQScdTFh/view?usp=drive_link)
- `best_age_model.pth`: [Download](https://drive.google.com/file/d/1NAM2FhkpTCEuqTrbnN6wtOHOZTmaRgjD/view?usp=drive_link)

---

## Folder Structure

notebooks/
├── eda/ # Exploratory Data Analysis & Feature Engineering
   ├── eda.ipynb # Full EDA peformed on the original dataset
   ├── eda_extra_data.ipynb # Task 1 EDA peformed on the original and extra datasets

├── task1/

  ├── task1_customCNN.ipynb # custom CNN model development for rice disease classification

  ├── task1_resNet50.ipynb # resNet50 model for disease classification

  └── task1_evaluation_model.ipynb # Comparison and final decision for task 1


├── task2/

  ├── task2_mobnetv2.ipynb # MobileNetV2 model for rice variety classification

  ├── task2_VGG16.ipynb # VGG16 model for variety classification

  └── task2_evaluation_model.ipynb # Comparison and final decision for task 2

├── task3/

  ├── task3_efNetb0.ipynb # EfficientNetB0 model for rice age regression

  ├── task3_inceptionV3_part1.ipynb # InceptionV3 training (part 1)

  ├── task3_inceptionV3_part2.ipynb # InceptionV3 training (part 2)

  └── task3_evaluation_model.ipynb # Model comparison and selection
  
└── final_prediction.ipynb # Run all 3 models on new data to output predictions


---

## Workflow Summary
This project tackles 3 tasks related to rice plant monitoring using deep learning models:
1. **EDA**
   - Input: Train and Test Dataset is given (placed at **Original data** folder in Google drive) 
   - Output: Preprocessed Train Dataset for each task (placed at **Processed data** folder in Google drive)   

2. **Task 1 – Disease Classification**
   - Output: One of 10 disease labels
   - Model: custom CNN model and resNet50 are implemented; resNet50 selected for best model

3. **Task 2 – Variety Classification**
   - Output: One of 10 rice variety classes
   - Models: MobileNetV2 and VGG16 are implemented; MobileNetV2 selected for best model 

4. **Task 3 – Age Regression**
   - Output: Estimated age of the plant in days
   - Models: InceptionV3 and EfficientNetB0; EfficientNetB0 selected for best model 

5. **Prediction**
   - Input: three best models and Test dataset 
   - Output: Predict submission file

---

## How to Review

All notebooks are implemented and tested in **Google Colab**. You can open each notebook directly in Colab and run each cell.

## Application

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
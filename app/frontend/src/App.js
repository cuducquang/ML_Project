// frontend/src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import { 
  Container, 
  Typography, 
  Button, 
  Card, 
  CardContent, 
  CardMedia,
  Grid,
  Paper
} from '@mui/material';
const PREDICT_API = "http://127.0.0.1:5000/predict_all"

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Reset previous predictions
      setPredictions(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setIsLoading(true);
    setPredictions(null);

    try {
      const response = await axios.post(PREDICT_API, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setPredictions(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const     renderPredictionCard = (title, predictionData) => {
    // Handle potential error scenarios
    if (predictionData.error) {
      return (
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 2, bgcolor: 'error.light', color: 'white' }}>
            <Typography variant="h6">{title}</Typography>
            <Typography variant="body2">
              {predictionData.error}
            </Typography>
          </Paper>
        </Grid>
      );
    }

    return (
      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">{title}</Typography>
          <Typography variant="body1">
            Predicted: {predictionData.predicted}
          </Typography>
          <Typography variant="body2">
            Confidence: {(predictionData.confidence * 100).toFixed(2)}%
          </Typography>
        </Paper>
      </Grid>
    );
  };

  return (
    <Container maxWidth="md" sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      mt: 4 
    }}>
      <Typography variant="h4" gutterBottom>
        Multi-Model Plant Predictor
      </Typography>

      {/* File Input */}
      <Button 
        variant="contained" 
        component="label" 
        sx={{ mb: 2 }}
      >
        Upload Plant Image
        <input 
          type="file" 
          hidden 
          accept="image/*"
          onChange={handleFileChange}
        />
      </Button>

      {/* Image Preview */}
      {previewImage && (
        <Card sx={{ maxWidth: 345, mb: 2 }}>
          <CardMedia
            component="img"
            height="240"
            image={previewImage}
            alt="Plant Image"
          />
        </Card>
      )}

      {/* Predict Button */}
      <Button 
        variant="contained" 
        color="primary" 
        onClick={handlePredict}
        disabled={!selectedFile || isLoading}
      >
        {isLoading ? 'Predicting...' : 'Run All Predictions'}
      </Button>

      {/* Prediction Results */}
      {predictions && (
        <Container sx={{ mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            Prediction Results
          </Typography>
          <Grid container spacing={2}>
            {renderPredictionCard('Variety Prediction', predictions.variety)}
            {renderPredictionCard('Disease Prediction', predictions.disease)}
            {/* {predictions.age_efficientnet && 
              renderPredictionCard('Age Prediction (EfficientNet)', predictions.age_efficientnet)} */}
          </Grid>
        </Container>
      )}
    </Container>
  );
}

export default App;
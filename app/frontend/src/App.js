import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  Button,
  Card,
  CardMedia,
  Grid,
  Paper,
  Box,
  CircularProgress,
} from "@mui/material";

const PREDICT_API = "https://rmit-ml-asm2.onrender.com/predict_all";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewImage(reader.result);
      };
      reader.readAsDataURL(file);
      setPredictions(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    setIsLoading(true);
    setPredictions(null);

    try {
      const response = await axios.post(PREDICT_API, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setPredictions(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Prediction failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const renderPredictionCard = (title, predictionData) => {
    if (!predictionData) return null;

    if (predictionData.error) {
      return (
        <Paper
          elevation={2}
          sx={{ p: 2, borderRadius: 2, bgcolor: "error.main", color: "white" }}
        >
          <Typography variant="h6">{title}</Typography>
          <Typography variant="body2">{predictionData.error}</Typography>
        </Paper>
      );
    }

    return (
      <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
        <Typography variant="h6">{title}</Typography>
        <Typography variant="body1">
          <strong>Predicted:</strong> {predictionData.predicted}
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          <strong>Confidence:</strong>{" "}
          {(predictionData.confidence * 100).toFixed(2)}%
        </Typography>
      </Paper>
    );
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        py: 4,
        minHeight: "100vh",
        backgroundColor: "#f0f2f5",
        display: "flex",
        justifyContent: "center",
        alignItems: "start",
      }}
    >
      <Paper elevation={4} sx={{ p: 4, borderRadius: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          🌾 Rice Plant Disease & Age Predictor
        </Typography>

        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 2,
          }}
        >
          {/* Upload Button */}
          <Button variant="contained" component="label" sx={{ mt: 2 }}>
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
            <Card sx={{ maxWidth: 345, mt: 2 }}>
              <CardMedia
                component="img"
                height="240"
                image={previewImage}
                alt="Uploaded Plant"
              />
            </Card>
          )}

          {/* Predict Button */}
          <Button
            variant="contained"
            color="primary"
            onClick={handlePredict}
            disabled={!selectedFile || isLoading}
            sx={{ mt: 2 }}
          >
            {isLoading ? (
              <>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                Predicting...
              </>
            ) : (
              "Run All Predictions"
            )}
          </Button>
        </Box>

        {/* Prediction Results */}
        {predictions && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              Prediction Results
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={4}>
                {renderPredictionCard("Variety", predictions.variety)}
              </Grid>
              <Grid item xs={12} sm={4}>
                {renderPredictionCard("Disease", predictions.disease)}
              </Grid>
              <Grid item xs={12} sm={4}>
                {renderPredictionCard("Age (days)", predictions.age)}
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default App;

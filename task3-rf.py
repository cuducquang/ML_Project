# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import hashlib
from collections import defaultdict
import time
from PIL import Image
from joblib import dump, load, Parallel, delayed

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Scikit-Image Imports
from skimage import color, feature, filters, measure, exposure, segmentation, restoration


def calculate_image_hash(image_path):
    """Calculate the hash of an image file."""
    hasher = hashlib.md5()
    with open(image_path, 'rb') as img_file:
        buf = img_file.read()
        hasher.update(buf)
    return hasher.hexdigest()


def find_duplicate_images(image_dir):
    """Find duplicate images in a directory with subdirectories."""
    hash_dict = defaultdict(list)

    # Traverse all subdirectories
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                img_hash = calculate_image_hash(file_path)
                hash_dict[img_hash].append(file_path)

    # Identify duplicates
    duplicates = {hash_val: paths for hash_val, paths in hash_dict.items() if len(paths) > 1}
    return duplicates


def extract_features(image_path, size=(128, 128)):
    """Extract advanced features from an image using scikit-image."""
    try:
        # Load and resize image
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize(size)
        img_array = np.array(img_resized)

        # Convert to different color spaces
        gray_img = color.rgb2gray(img_array)
        hsv_img = color.rgb2hsv(img_array)
        lab_img = color.rgb2lab(img_array)

        # Enhance image for better feature extraction
        gray_eq = exposure.equalize_adapthist(gray_img)

        # Extract color statistics features
        channels = {'red': img_array[:, :, 0], 'green': img_array[:, :, 1], 'blue': img_array[:, :, 2],
                    'hue': hsv_img[:, :, 0], 'saturation': hsv_img[:, :, 1], 'value': hsv_img[:, :, 2],
                    'l': lab_img[:, :, 0], 'a': lab_img[:, :, 1], 'b': lab_img[:, :, 2]}

        color_features = []
        for channel_name, channel in channels.items():
            color_features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])

        # Calculate vegetation indices
        r_mean, g_mean = np.mean(img_array[:, :, 0]), np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])

        # Green-Red Vegetation Index
        grvi = (g_mean - r_mean) / (g_mean + r_mean + 1e-6)

        # Normalized Difference Vegetation Index (NDVI-like)
        ndvi = (g_mean - r_mean) / (g_mean + r_mean + 1e-6)

        # Enhanced Vegetation Index
        evi = 2.5 * (g_mean - r_mean) / (g_mean + 6 * r_mean - 7.5 * b_mean + 1)

        # Texture features using local binary patterns
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray_eq, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, n_points + 2))
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize

        # Edge features
        edge_sobel = filters.sobel(gray_eq)
        edge_hist, _ = np.histogram(edge_sobel, bins=10)
        edge_hist = edge_hist / np.sum(edge_hist)

        # Segment the image to separate plant from background
        thresh = filters.threshold_otsu(gray_img)
        binary = gray_img < thresh  # For plant images, plants are usually darker

        # Clean up the binary image
        binary = measure.label(binary)

        # Compute region properties
        regions = measure.regionprops(binary)

        # Extract region features from the largest region (assuming it's the plant)
        region_features = []
        if regions:
            # Get the largest region by area
            largest_region = max(regions, key=lambda r: r.area)

            # Extract useful shape metrics
            region_features = [
                largest_region.area / (size[0] * size[1]),  # Normalized area
                largest_region.perimeter / (2 * (size[0] + size[1])),  # Normalized perimeter
                largest_region.eccentricity,
                largest_region.extent,
                largest_region.solidity,
                largest_region.euler_number
            ]
        else:
            # If no regions found, use defaults
            region_features = [0, 0, 0, 0, 0, 0]

        # Combine all features into a single vector
        all_features = np.concatenate([
            color_features,
            [grvi, ndvi, evi],
            lbp_hist,
            edge_hist,
            region_features
        ])

        return all_features

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def process_image(row, image_dir):
    """Process a single image for parallel execution."""
    image_path = os.path.join(image_dir, row['label'], f"{row['image_id']}")
    features = extract_features(image_path)
    return features, row['age']


def prepare_data_parallel(df, image_dir, n_jobs=-1):
    """Extract features in parallel for faster processing."""
    print(f"Extracting features from {len(df)} images using {n_jobs} jobs...")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_image)(row, image_dir) for _, row in df.iterrows()
    )

    # Filter out None results (failed extractions)
    valid_results = [r for r in results if r[0] is not None]

    # Separate features and labels
    features = [r[0] for r in valid_results]
    labels = [r[1] for r in valid_results]

    return np.array(features), np.array(labels)


def age_accuracy(true_values, predictions, tolerance=1):
    """
    Calculate 'accuracy' for age predictions within a tolerance
    """
    absolute_errors = np.abs(np.array(true_values) - np.array(predictions))
    correct_predictions = (absolute_errors <= tolerance).sum()
    return correct_predictions / len(true_values) * 100


def main():
    # Configuration options
    use_sample = True  # Use a smaller sample of data for testing
    sample_size = 200  # Number of images to use from each class
    do_grid_search = True  # Whether to perform grid search
    use_cached_features = False  # Whether to use cached features
    cache_file = "extracted_features.joblib"
    n_jobs = -1  # Number of parallel jobs (-1 for all cores)

    print("Starting Rice Plant Age Prediction...")

    # Load the metadata CSV
    meta_df = pd.read_csv("meta_train.csv")

    # Display the first 5 rows
    print("Dataset preview:")
    print(meta_df.head())

    # Check for nulls and duplicates
    print("\nData quality check:")
    print(f"Null values: {meta_df.isnull().sum().sum()}")
    print(f"Duplicates: {meta_df.duplicated('image_id').sum()}")

    # Basic data cleaning
    meta_df.dropna(subset=['age'], inplace=True)
    meta_df.drop_duplicates(subset='image_id', inplace=True)

    # Analyze age distribution
    print("\nAge statistics:")
    print(f"Min age: {meta_df['age'].min()} days")
    print(f"Max age: {meta_df['age'].max()} days")
    print(f"Mean age: {meta_df['age'].mean():.2f} days")
    print(f"Std dev: {meta_df['age'].std():.2f} days")

    # Image directory
    image_directory = "./train_images"

    # Use a sample for faster execution if requested
    if use_sample:
        # Get a balanced sample from each label
        sampled_df = pd.DataFrame()
        for label in meta_df['label'].unique():
            label_df = meta_df[meta_df['label'] == label]
            if len(label_df) > sample_size:
                label_sample = label_df.sample(sample_size, random_state=42)
                sampled_df = pd.concat([sampled_df, label_sample])
            else:
                sampled_df = pd.concat([sampled_df, label_df])

        meta_df = sampled_df
        print(f"\nUsing a balanced sample of {len(meta_df)} images")

    # Create data visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(meta_df['age'], bins=30, kde=True, color='lightgreen', edgecolor='black')
    plt.title("Age Distribution of Training Set")
    plt.xlabel("Age (days)")
    plt.ylabel("Frequency")
    plt.savefig("age_distribution.png")
    plt.close()

    # Split data into train and validation sets
    train_df, val_df = train_test_split(meta_df, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Extract features or load from cache
    if use_cached_features and os.path.exists(cache_file):
        print(f"\nLoading cached features from {cache_file}")
        cached_data = load(cache_file)
        X_train, y_train = cached_data['train']
        X_val, y_val = cached_data['val']
    else:
        print("\nExtracting image features...")
        start_time = time.time()

        # Extract features in parallel
        X_train, y_train = prepare_data_parallel(train_df, image_directory, n_jobs=n_jobs)
        X_val, y_val = prepare_data_parallel(val_df, image_directory, n_jobs=n_jobs)

        # Cache the extracted features
        dump({'train': (X_train, y_train), 'val': (X_val, y_val)}, cache_file)

        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

    print(f"Feature shape: {X_train.shape}")

    # Create a pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    if do_grid_search:
        # Parameter grid for RandomForest
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        print("\nStarting grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3,
            scoring='neg_mean_absolute_error',
            verbose=1, n_jobs=n_jobs
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        print(f"Grid search completed in {training_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")

        # Use the best model
        model = grid_search.best_estimator_

        # Save the model
        dump(model, 'best_rf_model.joblib')
    else:
        # Use default parameters
        print("\nTraining with default parameters...")
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=n_jobs,
                random_state=42
            ))
        ])

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save the model
        dump(model, 'default_rf_model.joblib')

    # Make predictions on validation set
    y_pred = model.predict(X_val)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    acc_1day = age_accuracy(y_val, y_pred, tolerance=1)
    acc_3day = age_accuracy(y_val, y_pred, tolerance=3)
    acc_5day = age_accuracy(y_val, y_pred, tolerance=5)

    # Print metrics
    print("\nValidation Metrics:")
    print(f"Mean Absolute Error: {mae:.2f} days")
    print(f"Root Mean Squared Error: {rmse:.2f} days")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy (±1 day): {acc_1day:.2f}%")
    print(f"Accuracy (±3 days): {acc_3day:.2f}%")
    print(f"Accuracy (±5 days): {acc_5day:.2f}%")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--')
    plt.xlabel('Actual Age (days)')
    plt.ylabel('Predicted Age (days)')
    plt.title('Actual vs Predicted Age')
    plt.grid(True)
    plt.savefig("actual_vs_predicted.png")
    plt.close()

    # Plot error distribution
    errors = y_pred - y_val
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error (days)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.savefig("error_distribution.png")
    plt.close()

    # Plot feature importances
    if hasattr(model.named_steps['rf'], 'feature_importances_'):
        feature_importances = model.named_steps['rf'].feature_importances_

        # Get the top 20 features
        top_n = min(20, len(feature_importances))
        indices = np.argsort(feature_importances)[-top_n:]

        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), feature_importances[indices])
        plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

    print("\nResults and visualizations saved.")


if __name__ == "__main__":
    main()
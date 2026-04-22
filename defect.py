import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel, gabor
import matplotlib.pyplot as plt
from collections import Counter
import random

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (128, 128)
path = r"C:\Users\admin\Downloads\carpet\carpet\test"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =========================
# FEATURE EXTRACTION FUNCTIONS
# =========================
def extract_texture_features(image):
    """Extract texture features using LBP, GLCM, and Gabor filters"""
    features = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. LBP (Local Binary Pattern) features
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist / lbp_hist.sum()
    features.extend(lbp_hist)
    
    # 2. GLCM features
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    # Extract GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # 3. Edge features using Sobel
    edge_sobel = sobel(gray)
    edge_density = np.mean(edge_sobel)
    edge_std = np.std(edge_sobel)
    features.extend([edge_density, edge_std])
    
    # 4. Gabor filter features
    gabor_features = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for frequency in [0.1, 0.3, 0.5]:
            gabor_real, gabor_imag = gabor(gray, frequency=frequency, theta=theta)
            gabor_features.append(np.mean(gabor_real))
            gabor_features.append(np.std(gabor_real))
    
    features.extend(gabor_features[:10])  # Take first 10 to avoid too many features
    
    # 5. Statistical features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.percentile(gray, 25),
        np.percentile(gray, 75),
        np.max(gray) - np.min(gray)  # Range
    ])
    
    return np.array(features)

def extract_color_features(image):
    """Extract color-based features"""
    features = []
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # RGB histograms
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = hist / hist.sum()
        features.extend(hist.flatten()[:16])  # Take first 16 bins
    
    # HSV statistics
    for i in range(3):
        features.extend([
            np.mean(hsv[:,:,i]),
            np.std(hsv[:,:,i]),
            np.percentile(hsv[:,:,i], 50)
        ])
    
    # LAB statistics
    for i in range(3):
        features.extend([
            np.mean(lab[:,:,i]),
            np.std(lab[:,:,i])
        ])
    
    return np.array(features)

def extract_defect_features(image):
    """Extract defect-specific features using morphological operations"""
    features = []
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply multiple thresholding methods
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combine thresholds
    combined_thresh = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contour features
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        features.extend([
            len(contours),  # Number of defects
            np.mean(areas) if areas else 0,
            np.std(areas) if len(areas) > 1 else 0,
            np.max(areas) if areas else 0,
            np.sum(areas),  # Total defect area
            np.mean(perimeters) if perimeters else 0,
        ])
        
        # Defect area ratio
        defect_ratio = np.sum(areas) / (gray.shape[0] * gray.shape[1])
        features.append(defect_ratio)
    else:
        features.extend([0, 0, 0, 0, 0, 0, 0])
    
    # Texture uniformity (variance)
    features.append(np.var(gray))
    
    return np.array(features)

def extract_all_features(image_path):
    """Extract all features from an image"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        
        # Extract feature sets
        texture_feats = extract_texture_features(img)
        color_feats = extract_color_features(img)
        defect_feats = extract_defect_features(img)
        
        # Combine all features
        all_features = np.concatenate([texture_feats, color_feats, defect_feats])
        
        return all_features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# =========================
# LOAD DATA AND EXTRACT FEATURES
# =========================
def load_dataset(base_path):
    """Load dataset and extract features"""
    good_paths = []
    defect_paths = []
    
    # Collect image paths
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                if "good" in root.lower():
                    good_paths.append(img_path)
                else:
                    defect_paths.append(img_path)
    
    print(f"Found {len(good_paths)} good images")
    print(f"Found {len(defect_paths)} defect images")
    
    # Balance dataset (take equal number from each class)
    min_count = min(len(good_paths), len(defect_paths))
    good_paths = random.sample(good_paths, min_count)
    defect_paths = random.sample(defect_paths, min_count)
    
    print(f"\nUsing {len(good_paths)} images from each class")
    
    # Extract features
    features = []
    labels = []
    
    print("\nExtracting features from good images...")
    for img_path in good_paths:
        feats = extract_all_features(img_path)
        if feats is not None:
            features.append(feats)
            labels.append(0)
    
    print("Extracting features from defect images...")
    for img_path in defect_paths:
        feats = extract_all_features(img_path)
        if feats is not None:
            features.append(feats)
            labels.append(1)
    
    return np.array(features), np.array(labels), good_paths, defect_paths

# Load and extract features
print("="*50)
print("Loading dataset and extracting features...")
print("="*50)

X, y, good_paths, defect_paths = load_dataset(path)

print(f"\nTotal samples: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")
print(f"Class distribution: {dict(Counter(y))}")

# Check for NaN or infinite values
print(f"\nChecking for invalid values...")
print(f"NaN values: {np.isnan(X).sum()}")
print(f"Infinite values: {np.isinf(X).sum()}")

# Replace NaN and infinite values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# =========================
# TRAIN RANDOM FOREST
# =========================
print("\n" + "="*50)
print("Training Random Forest Classifier...")
print("="*50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with hyperparameter tuning
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)

# Train
rf.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Predictions
y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

# Training accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

# Test accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed metrics
print("\n" + "="*50)
print("Classification Report (Test Set):")
print("="*50)
print(classification_report(y_test, y_pred_test, target_names=['Good', 'Defect']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Metrics:")
print(f"True Negatives (Good correctly identified): {tn}")
print(f"False Positives (Good identified as Defect): {fp}")
print(f"False Negatives (Defect identified as Good): {fn}")
print(f"True Positives (Defect correctly identified): {tp}")
print(f"Good Class Accuracy: {tn/(tn+fp)*100:.2f}%" if (tn+fp) > 0 else "N/A")
print(f"Defect Class Accuracy: {tp/(tp+fn)*100:.2f}%")
print(f"Overall Accuracy: {(tn+tp)/(tn+tp+fn+fp)*100:.2f}%")

# Feature importance
feature_importance = rf.feature_importances_
print(f"\nTop 10 most important features:")
top_indices = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")

# =========================
# SAVE MODEL
# =========================
import joblib
joblib.dump(rf, 'carpet_defect_rf_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("\n✅ Random Forest model saved as 'carpet_defect_rf_model.pkl'")
print("✅ Scaler saved as 'feature_scaler.pkl'")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_defect(image_path, model, scaler):
    """Predict defect using trained model"""
    features = extract_all_features(image_path)
    if features is None:
        return "Error: Could not process image", 0.0
    
    features = np.nan_to_num(features.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    result = "Defect" if prediction == 1 else "Good"
    confidence = max(probability)
    
    return result, confidence

# Test on sample images
print("\n" + "="*50)
print("Sample Predictions on Test Set:")
print("="*50)

# Get some test images
test_good_paths = [good_paths[i] for i in range(min(3, len(good_paths)))]
test_defect_paths = [defect_paths[i] for i in range(min(3, len(defect_paths)))]

print("\nTesting Good Images:")
for img_path in test_good_paths:
    pred, conf = predict_defect(img_path, rf, scaler)
    print(f"  {os.path.basename(img_path)}: {pred} (confidence: {conf:.2%})")

print("\nTesting Defect Images:")
for img_path in test_defect_paths:
    pred, conf = predict_defect(img_path, rf, scaler)
    print(f"  {os.path.basename(img_path)}: {pred} (confidence: {conf:.2%})")

# =========================
# VISUALIZATION
# =========================
# Plot confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix heatmap
im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Good', 'Defect'])
axes[0].set_yticklabels(['Good', 'Defect'])

# Add text annotations
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

# Feature importance bar chart
top_features = feature_importance[top_indices]
axes[1].barh(range(len(top_features)), top_features)
axes[1].set_title('Top 10 Feature Importances')
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('Feature Index')

plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Training complete! The model is now ready for use.")
print(f"Expected accuracy: {test_accuracy*100:.1f}%")
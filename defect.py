import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel, gabor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import joblib
from tqdm import tqdm
import pickle
import hashlib
import time

print("="*70)
print(" FABRIC DEFECT DETECTION SYSTEM - COMPLETE TRAINING")
print("="*70)

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (128, 128)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Update this path to your dataset location
DATASET_PATH = r"C:\Users\admin\Documents\Defect_Dataset"
CACHE_FILE = "extracted_features_cache.pkl"

print(f"\n📁 Dataset Path: {DATASET_PATH}")
print(f"💾 Cache File: {CACHE_FILE}")

# =========================
# COMPLETE FIXED SEVERITY CALCULATION WITH BETTER DISTRIBUTION
def calculate_severity(image):
    """Severity based on gradient magnitude (color-invariant)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 1. Compute gradient (edge map) – ignores absolute color
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6) * 255
    gradient = gradient.astype(np.uint8)

    # 2. Threshold on gradient (not on gray)
    _, thresh_otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    combined = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    areas = [cv2.contourArea(c) for c in contours]
    total_area = gray.shape[0] * gray.shape[1]
    area_ratio = sum(areas) / total_area
    num_defects = len(contours)
    num_defects_penalty = min(num_defects / 20, 0.3)
    avg_defect_size = np.mean(areas) if areas else 0
    size_penalty = min(avg_defect_size / 5000, 0.3)

    severity_score = (area_ratio * 0.6) + (num_defects_penalty * 0.2) + (size_penalty * 0.2)

    if severity_score < 0.15:
        return 0
    elif severity_score < 0.4:
        return 1
    else:
        return 2
def calculate_severity_color_invariant(image):
    """Improved version - completely color invariant"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Use gradient magnitude (edge detection) - this ignores absolute color values
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6) * 255
    gradient = gradient.astype(np.uint8)
    
    # Apply CLAHE to normalize local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gradient_norm = clahe.apply(gradient)
    
    # Multiple thresholding methods on gradient
    _, thresh_otsu = cv2.threshold(gradient_norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_adaptive = cv2.adaptiveThreshold(gradient_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    combined = cv2.bitwise_or(thresh_otsu, thresh_adaptive)
    
    # Clean up
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    # Calculate severity based on structural anomalies
    areas = [cv2.contourArea(c) for c in contours]
    total_area = gray.shape[0] * gray.shape[1]
    area_ratio = sum(areas) / total_area
    
    # Count only significant defects (ignore tiny noise)
    significant_defects = [a for a in areas if a > 50]
    num_significant = len(significant_defects)
    
    # Severity based on structural defects
    if num_significant == 0:
        return 0
    elif num_significant <= 3 and area_ratio < 0.1:
        return 0  # Low
    elif num_significant <= 6 or area_ratio < 0.25:
        return 1  # Medium
    else:
        return 2  # High


# =========================
# FEATURE EXTRACTION - 89 FIXED FEATURES
# =========================
def extract_features_89(image):
    """
    Extract exactly 89 features - MUST match UI
    Features: 9 stats + 16 hist + 16 LBP + 8 GLCM + 6 edge + 12 Gabor + 10 defect + 12 color = 89
    """
    features = []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. Basic statistical features (9 features)
    features.extend([
        np.mean(gray),           # Mean intensity
        np.std(gray),            # Standard deviation
        np.median(gray),         # Median intensity
        np.percentile(gray, 25), # 25th percentile
        np.percentile(gray, 75), # 75th percentile
        np.percentile(gray, 90), # 90th percentile
        np.percentile(gray, 10), # 10th percentile
        np.max(gray) - np.min(gray),  # Intensity range
        np.var(gray)             # Variance
    ])
    
    # 2. Histogram features (16 features)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    hist = hist / (hist.sum() + 1e-6)  # Normalize
    features.extend(hist.flatten())
    
    # 3. LBP (Local Binary Pattern) features (16 features)
    for radius, n_points in [(1, 8), (2, 16)]:
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        hist_padded = np.zeros(8)
        hist_padded[:min(8, len(hist))] = hist[:8]
        features.extend(hist_padded)
    
    # 4. GLCM (Gray Level Co-occurrence Matrix) features (8 features)
    try:
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2], 256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
            prop_vals = graycoprops(glcm, prop)
            features.append(np.mean(prop_vals))
            features.append(np.std(prop_vals))
    except:
        features.extend([0] * 8)
    
    # 5. Edge detection features (6 features)
    edges_sobel = sobel(gray)
    edges_canny = cv2.Canny(gray, 50, 150)
    features.extend([
        np.mean(edges_sobel),                    # Mean edge intensity
        np.std(edges_sobel),                     # Edge variation
        np.max(edges_sobel),                     # Maximum edge intensity
        np.sum(edges_canny > 0) / (gray.shape[0] * gray.shape[1]),  # Edge density
        np.percentile(edges_sobel, 75),          # 75th percentile
        np.percentile(edges_sobel, 90)           # 90th percentile
    ])
    
    # 6. Gabor filter features (12 features)
    gabor_features = []
    try:
        for theta in [0, np.pi/4]:
            for frequency in [0.1, 0.2, 0.3]:
                gabor_real, _ = gabor(gray, frequency=frequency, theta=theta)
                gabor_features.append(np.mean(gabor_real))
                gabor_features.append(np.std(gabor_real))
    except:
        pass
    
    # Ensure exactly 12 features
    while len(gabor_features) < 12:
        gabor_features.append(0)
    features.extend(gabor_features[:12])
    
    # 7. Defect-specific morphological features (10 features)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    combined_thresh = cv2.bitwise_or(thresh1, thresh2)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        features.extend([
            min(len(contours), 5),           # Number of defects (capped)
            np.mean(areas) if areas else 0,   # Average defect area
            np.std(areas) if len(areas) > 1 else 0,  # Area variation
            np.max(areas) if areas else 0,    # Maximum defect area
            np.sum(areas),                    # Total defect area
            np.sum(areas) / (gray.shape[0] * gray.shape[1])  # Defect area ratio
        ])
        features.extend([0, 0, 0, 0])  # Pad to 10 features
    else:
        features.extend([0] * 10)
    
    # 8. Color features (12 features)
    if len(image.shape) == 3:
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for channel in range(3):
                features.extend([
                    np.mean(hsv[:,:,channel]),      # Mean
                    np.std(hsv[:,:,channel]),       # Standard deviation
                    np.median(hsv[:,:,channel]),    # Median
                    np.percentile(hsv[:,:,channel], 25)  # 25th percentile
                ])
        except:
            features.extend([0] * 12)
    else:
        features.extend([0] * 12)
    
    return np.array(features, dtype=np.float32)

# =========================
# LOAD DATASET
# =========================
def load_dataset(dataset_path):
    """Load all images from the dataset directory"""
    images = []
    labels = []
    file_paths = []
    
    print("\n📂 Scanning dataset directory...")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                img_path = os.path.join(root, file)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, IMG_SIZE)
                    
                    folder_name = os.path.basename(root).lower()
                    
                    # Determine label based on folder name
                    if any(keyword in folder_name for keyword in ['good', 'ok', 'normal', 'defect free', 'ok_image']):
                        labels.append(0)  # Good
                    else:
                        labels.append(1)  # Defect
                    
                    images.append(img)
                    file_paths.append(img_path)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {img_path}: {e}")
                    continue
    
    return np.array(images), np.array(labels), file_paths

# =========================
# CREATE SYNTHETIC GOOD IMAGES
# =========================
def create_synthetic_good_images(defect_images, num_needed):
    """Create synthetic good images by processing defect images"""
    synthetic_good = []
    
    print("🎨 Creating synthetic good images...")
    
    for i in range(min(num_needed, len(defect_images) * 2)):
        img = defect_images[i % len(defect_images)].copy()
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Apply median blur to smooth out defects
        smoothed = cv2.medianBlur(gray, 5)
        
        # Apply inpainting to remove defects
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        inpainted = cv2.inpaint(smoothed, thresh, 3, cv2.INPAINT_TELEA)
        
        synthetic_good.append(cv2.cvtColor(inpainted, cv2.COLOR_GRAY2RGB))
        
        if len(synthetic_good) >= num_needed:
            break
    
    print(f"✅ Created {len(synthetic_good)} synthetic good images")
    return synthetic_good[:num_needed]

# =========================
# GET DATASET HASH FOR CACHE VALIDATION
# =========================
def get_dataset_hash():
    """Generate hash of dataset for cache validation"""
    if not os.path.exists(DATASET_PATH):
        return None
    
    image_files = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                img_path = os.path.join(root, file)
                if os.path.exists(img_path):
                    image_files.append((img_path, os.path.getmtime(img_path)))
    
    hash_input = str(sorted(image_files))
    return hashlib.md5(hash_input.encode()).hexdigest()

# =========================
# MAIN TRAINING PIPELINE
# =========================
# Try to load from cache
cache_valid = False
if os.path.exists(CACHE_FILE):
    try:
        print("\n🔍 Checking cache...")
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        current_hash = get_dataset_hash()
        if current_hash and cache_data.get('dataset_hash') == current_hash:
            cache_valid = True
            print("✅ Found valid cached features! Loading...")
            X = cache_data['X']
            y = cache_data['y']
            severities = cache_data['severities']
            print(f"📊 Loaded {len(X)} samples from cache")
            print(f"🔬 Feature dimension: {X.shape[1]}")
            print(f"📈 Class distribution: {dict(Counter(y))}")
        else:
            print("⚠️ Cache is outdated. Extracting features again...")
    except Exception as e:
        print(f"⚠️ Error loading cache: {e}")

if not cache_valid:
    print("\n" + "="*70)
    print(" STEP 1: LOADING DATASET")
    print("="*70)
    
    # Load dataset
    X_images, y, file_paths = load_dataset(DATASET_PATH)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images found: {len(X_images)}")
    print(f"   Class distribution: {dict(Counter(y))}")
    
    if len(X_images) == 0:
        print("\n❌ No images found! Please check your dataset path.")
        print(f"   Current path: {DATASET_PATH}")
        exit()
    
    # Handle missing classes
    if len(np.unique(y)) < 2:
        print("\n⚠️ Dataset has only one class! Creating synthetic data...")
        
        if 0 not in y:  # No good images
            print("   → No good images found. Creating synthetic good images...")
            defect_images = X_images[y == 1]
            num_good_needed = len(defect_images) // 2
            synthetic_good = create_synthetic_good_images(defect_images, num_good_needed)
            
            X_images = np.concatenate([X_images, np.array(synthetic_good)])
            y = np.concatenate([y, np.zeros(len(synthetic_good))])
            
        elif 1 not in y:  # No defect images
            print("   → No defect images found. Creating synthetic defect images...")
            good_images = X_images[y == 0]
            num_defect_needed = len(good_images) // 2
            
            synthetic_defect = []
            for i in range(min(num_defect_needed, len(good_images))):
                img = good_images[i % len(good_images)].copy()
                h, w = img.shape[:2]
                cv2.circle(img, (w//2, h//2), 20, (0, 0, 255), -1)
                synthetic_defect.append(img)
            
            X_images = np.concatenate([X_images, np.array(synthetic_defect)])
            y = np.concatenate([y, np.ones(len(synthetic_defect))])
        
        print(f"   After synthetic creation: {dict(Counter(y))}")
    
    # Balance the dataset
    min_class_count = min(Counter(y).values())
    print(f"\n⚖️ Balancing dataset to {min_class_count} samples per class...")
    
    balanced_X = []
    balanced_y = []
    
    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        selected_indices = np.random.choice(class_indices, min_class_count, replace=False)
        balanced_X.extend(X_images[selected_indices])
        balanced_y.extend(y[selected_indices])
    
    X_images = np.array(balanced_X)
    y = np.array(balanced_y)
    
    print(f"   Balanced dataset: {len(X_images)} images")
    print(f"   Class distribution: {dict(Counter(y))}")
    
    # Apply data augmentation
    print("\n" + "="*70)
    print(" STEP 2: DATA AUGMENTATION")
    print("="*70)
    
    augmented_X = []
    augmented_y = []
    
    for img, label in tqdm(zip(X_images, y), total=len(X_images), desc="🔄 Augmenting images"):
        # Add original
        augmented_X.append(img)
        augmented_y.append(label)
        
        # Add augmented versions for defect images
        if label == 1:
            # Horizontal flip
            augmented_X.append(cv2.flip(img, 1))
            augmented_y.append(label)
            
            # Slight rotation
            if random.random() > 0.7:
                angle = random.uniform(-10, 10)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                rotated = cv2.warpAffine(img, M, (w, h))
                augmented_X.append(rotated)
                augmented_y.append(label)
    
    X_images = np.array(augmented_X)
    y = np.array(augmented_y)
    
    print(f"\n📊 After augmentation:")
    print(f"   Total images: {len(X_images)}")
    print(f"   Class distribution: {dict(Counter(y))}")
    
    # Calculate severities
    print("\n" + "="*70)
    print(" STEP 3: CALCULATING SEVERITY LEVELS")
    print("="*70)
    
    severities = []
    for img, label in tqdm(zip(X_images, y), total=len(X_images), desc="📊 Calculating severity"):
        if label == 1:
            severity = calculate_severity_color_invariant(img)
            severities.append(severity)
        else:
            severities.append(0)
    
    severities = np.array(severities)
    
    print(f"\n📊 Severity distribution (defects only):")
    sev_dist = Counter(severities[y==1])
    total_defects = len(severities[y==1])
    print(f"   Low (0): {sev_dist.get(0, 0)} ({sev_dist.get(0, 0)/total_defects*100:.1f}%)")
    print(f"   Medium (1): {sev_dist.get(1, 0)} ({sev_dist.get(1, 0)/total_defects*100:.1f}%)")
    print(f"   High (2): {sev_dist.get(2, 0)} ({sev_dist.get(2, 0)/total_defects*100:.1f}%)")
    
    # Extract features
    print("\n" + "="*70)
    print(" STEP 4: EXTRACTING FEATURES (89 DIMENSIONS)")
    print("="*70)
    
    features_list = []
    for img in tqdm(X_images, desc="🔬 Extracting features"):
        feats = extract_features_89(img)
        features_list.append(feats)
    
    X = np.array(features_list, dtype=np.float32)
    print(f"\n✅ Feature extraction complete!")
    print(f"   Feature vector dimension: {X.shape[1]} (should be 89)")
    
    # Save to cache
    cache_data = {
        'dataset_hash': get_dataset_hash(),
        'X': X,
        'y': y,
        'severities': severities,
        'num_samples': len(X),
        'feature_dim': X.shape[1]
    }
    
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"\n💾 Features cached to {CACHE_FILE}")

# Split data
print("\n" + "="*70)
print(" STEP 5: SPLITTING DATA")
print("="*70)

X_train, X_test, y_train, y_test, sev_train, sev_test = train_test_split(
    X, y, severities, test_size=0.2, random_state=SEED, stratify=y
)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")
print(f"📈 Train distribution: {dict(Counter(y_train))}")
print(f"📈 Test distribution: {dict(Counter(y_test))}")

# Scale features
print("\n" + "="*70)
print(" STEP 6: SCALING FEATURES")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# =========================
# TRAIN BINARY CLASSIFIER
# =========================
print("\n" + "="*70)
print(" STEP 7: TRAINING BINARY CLASSIFIER")
print("="*70)

binary_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=0
)

print("🏋️ Training Random Forest classifier...")
binary_model.fit(X_train_scaled, y_train)

# Cross-validation
print("\n📊 Performing cross-validation...")
cv_scores = cross_val_score(binary_model, X_train_scaled, y_train, cv=5)
print(f"   Cross-validation scores: {cv_scores}")
print(f"   Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluation
print("\n" + "="*70)
print(" STEP 8: EVALUATING BINARY CLASSIFIER")
print("="*70)

y_pred = binary_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Good', 'Defect']))

cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\n📈 Detailed Metrics:")
print(f"   ✅ Good correctly identified: {tn}")
print(f"   ❌ Good misidentified as defect: {fp}")
print(f"   ❌ Defect misidentified as good: {fn}")
print(f"   ✅ Defect correctly identified: {tp}")
print(f"   Good Class Accuracy: {tn/(tn+fp)*100:.2f}%" if (tn+fp) > 0 else "N/A")
print(f"   Defect Class Accuracy: {tp/(tp+fn)*100:.2f}%")

# =========================
# TRAIN SEVERITY CLASSIFIER
# =========================
print("\n" + "="*70)
print(" STEP 9: TRAINING SEVERITY CLASSIFIER")
print("="*70)

severity_model = None

defect_train_idx = y_train == 1
X_defect_train = X_train_scaled[defect_train_idx]
sev_train_defect = sev_train[defect_train_idx]

print(f"📊 Defect samples for severity training: {len(X_defect_train)}")
print(f"📈 Severity distribution in training: {dict(Counter(sev_train_defect))}")

if len(X_defect_train) > 0 and len(np.unique(sev_train_defect)) >= 2:
    print("🏋️ Training severity classifier...")
    
    severity_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    )
    
    severity_model.fit(X_defect_train, sev_train_defect)
    
    # Evaluate severity
    defect_test_idx = y_test == 1
    X_defect_test = X_test_scaled[defect_test_idx]
    sev_test_defect = sev_test[defect_test_idx]
    
    # Evaluate severity only if we have at least 2 classes in test set
if len(X_defect_test) > 0 and len(np.unique(sev_test_defect)) >= 2:
    sev_pred = severity_model.predict(X_defect_test)
    sev_accuracy = accuracy_score(sev_test_defect, sev_pred)
    print(f"\n🎯 Severity Classification Accuracy: {sev_accuracy:.4f} ({sev_accuracy*100:.2f}%)")
    
    # Get unique classes present in test set
    unique_classes = np.unique(sev_test_defect)
    # Create target names only for those classes
    severity_names = ['Low', 'Medium', 'High']
    present_names = [severity_names[i] for i in unique_classes]
    
    print("\n📋 Severity Classification Report:")
    print(classification_report(
        sev_test_defect, 
        sev_pred,
        labels=unique_classes,          # Use only classes that actually appear
        target_names=present_names,     # Match names to those classes
        zero_division=0
    ))
else:
    print("⚠️ Not enough severity classes in test set for detailed report")
# =========================
# SAVE MODELS
# =========================
print("\n" + "="*70)
print(" STEP 10: SAVING MODELS")
print("="*70)

joblib.dump(binary_model, 'binary_classifier.pkl')
print("✅ Binary classifier saved as 'binary_classifier.pkl'")

if severity_model:
    joblib.dump(severity_model, 'severity_classifier.pkl')
    print("✅ Severity classifier saved as 'severity_classifier.pkl'")
else:
    print("⚠️ Severity classifier not saved (insufficient data)")

joblib.dump(scaler, 'feature_scaler.pkl')
print("✅ Feature scaler saved as 'feature_scaler.pkl'")

# =========================
# FEATURE IMPORTANCE
# =========================
print("\n" + "="*70)
print(" STEP 11: FEATURE IMPORTANCE ANALYSIS")
print("="*70)

importances = binary_model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

print("🏆 Top 10 most important features:")
for i, idx in enumerate(top_indices, 1):
    print(f"   {i}. Feature {idx}: {importances[idx]:.6f}")

# =========================
# VISUALIZATION
# =========================
print("\n" + "="*70)
print(" STEP 12: GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
axes[0, 0].set_title('Binary Classification Confusion Matrix', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# 2. Feature Importance
axes[0, 1].barh(range(10), importances[top_indices])
axes[0, 1].set_title('Top 10 Feature Importances', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_ylabel('Feature Index')

# 3. Class Distribution
class_dist = Counter(y)
colors = ['#2ecc71', '#e74c3c']
bars = axes[1, 0].bar(class_dist.keys(), class_dist.values(), color=colors)
axes[1, 0].set_title('Final Class Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(['Good', 'Defect'])
for bar, count in zip(bars, class_dist.values()):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

# 4. Severity Distribution
if severity_model and 'sev_test_defect' in locals() and len(sev_test_defect) > 0:
    sev_dist_test = Counter(sev_test_defect)
    colors_sev = ['#2ecc71', '#f39c12', '#e74c3c']
    bars_sev = axes[1, 1].bar(sev_dist_test.keys(), sev_dist_test.values(), color=colors_sev)
    axes[1, 1].set_title('Severity Distribution in Test Set', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Severity Level')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Low', 'Medium', 'High'])
    for bar, count in zip(bars_sev, sev_dist_test.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
else:
    axes[1, 1].text(0.5, 0.5, 'Severity Data Not Available\n(Insufficient defect samples)',
                   ha='center', va='center', transform=axes[1, 1].transAxes,
                   fontsize=12, style='italic')
    axes[1, 1].set_title('Severity Distribution', fontsize=12, fontweight='bold')

plt.suptitle('Fabric Defect Detection System - Training Results', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved as 'training_results.png'")

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "="*70)
print(" ✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Final Results:")
print(f"   ✅ Binary Classification Accuracy: {accuracy:.2%}")
print(f"   📈 Cross-validation Score: {cv_scores.mean():.2%}")
print(f"   📊 Total Training Samples: {len(X_train)}")
print(f"   🔬 Features Extracted: {X.shape[1]}")

if severity_model and 'sev_accuracy' in locals():
    print(f"   🎯 Severity Classification Accuracy: {sev_accuracy:.2%}")

print("\n📁 Saved Files:")
print("   📄 binary_classifier.pkl - Binary classification model")
if severity_model:
    print("   📄 severity_classifier.pkl - Severity classification model")
print("   📄 feature_scaler.pkl - Feature scaling model")
print(f"   📄 {CACHE_FILE} - Cached features (for faster loading)")
print("   📄 training_results.png - Visualization of results")

print("\n💡 Next Steps:")
print("   1. Run the Streamlit UI: streamlit run complete_ui.py")
print("   2. Upload images for defect detection")
print("   3. Test with video files")
print("   4. Use live camera for real-time detection")

print("\n" + "="*70)
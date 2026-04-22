import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIG
# =========================
IMG_SIZE = (128, 128)
path = r"C:\Users\admin\Downloads\carpet\carpet\test"

# =========================
# LOAD DATA
# =========================
def load_data(base_path):
    images = []
    labels = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)

                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, IMG_SIZE)
                    img = img / 255.0
                    images.append(img)

                    if "good" in root.lower():
                        labels.append(0)
                    else:
                        labels.append(1)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    print("Loaded images:", len(images))
    return np.array(images), np.array(labels, dtype=np.int32)

x, y = load_data(path)

print("Total images:", len(x))
print("Label distribution:", np.bincount(y))

if len(x) == 0:
    print("No images loaded")
    exit()

# =========================
# SPLIT
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# =========================
# AUGMENTATION (slightly improved)
# =========================
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# BINARY MODEL
# =========================
model_binary = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# ✅ FIX: learning rate inside optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model_binary.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ FIX: better early stopping
early_stop = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=3,         # Epochs to wait for improvement
    verbose=1,           # Log to console
    mode='min',          # Minimize the metric
    restore_best_weights=True # Revert to best model
)


model_binary.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    callbacks=[early_stop]
)

# =========================
# SEVERITY FUNCTION
# =========================
def get_severity(image):
    img_255 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    defect_area = np.count_nonzero(thresh)
    total_area = thresh.size
    ratio = defect_area / total_area

    if ratio < 0.55:
        return 0
    elif ratio < 0.90:
        return 1
    else:
        return 2

# =========================
# PREPARE SEVERITY DATA
# =========================
x_defect = []
y_severity = []

for img, label in zip(x_train, y_train):
    if label == 1:
        x_defect.append(img)
        y_severity.append(get_severity(img))

x_defect = np.array(x_defect)
y_severity = np.array(y_severity, dtype=np.int32)

print("Defect samples:", len(x_defect))
print("Severity distribution:", np.bincount(y_severity))

import numpy as np
unique, counts = np.unique(y_severity, return_counts=True)
print(dict(zip(unique, counts)))

ratios = []
for img in x_defect[:10]: # Check first 10 images
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ratios.append(np.sum(thresh > 0) / thresh.size)

print(f"Sample Ratios: {ratios}")
print(f"Min Ratio: {min(ratios)}, Max Ratio: {max(ratios)}")

# =========================
# 8. Stage 2 Model
# =========================
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# Data Augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])


# =========================
# SEVERITY MODEL
# =========================
if len(x_defect) > 0:

    model_severity = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        data_augmentation,

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model_severity.compile(
        optimizer=optimizer2,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop2 = EarlyStopping(
        monitor='val_loss',
        patience=0,
        restore_best_weights=True
    )

    model_severity.fit(
        x_defect, y_severity,
        epochs=10,
        validation_split=0.2,
        callbacks=[early_stop2]
    )

else:
    print("No defect data for severity model")

# =========================
# PREDICTION
# =========================
def predict(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model_binary.predict(img)[0][0]

    # ✅ Slightly improved threshold
    if prob < 0.4:
        return "Non-defect"

    if len(x_defect) == 0:
        return "Defect (severity unavailable)"

    severity = model_severity.predict(img)
    idx = np.argmax(severity)

    labels = ["Low", "Moderate", "High"]
    return f"Defect - {labels[idx]}"

# =========================
# TEST
# =========================
sample_image = x_test[0]
cv2.imwrite("sample.jpg", (sample_image * 255).astype(np.uint8))

print("Prediction:", predict("sample.jpg"))
# =========================
# SAVE MODELS
# =========================
model_binary.save("binary_model.keras")

if len(x_defect) > 0:
    model_severity.save("severity_model.keras")

print("✅ Models saved successfully!")
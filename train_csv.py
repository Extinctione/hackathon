import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load CSV (28x28 RGB images)
data = pd.read_csv("hmnist_28_28_RGB.csv")

# Split into features and labels
X = data.drop("label", axis=1).values
y = data["label"].values

# Normalize and reshape
X = X / 255.0
X = X.reshape(-1, 28, 28, 3)

# Binary classification: malignant (mel=4) vs benign (others)
y_binary = np.where(y == 4, 1, 0)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_binary, test_size=0.2, stratify=y_binary
)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Save model
model.save("cancer_model.h5")
print("✅ Model trained and saved as cancer_model.h5")

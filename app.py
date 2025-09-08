import os
import shutil
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
RAW_DATA = "dataset"              # your raw images
SPLIT_DIR = "skin_cancer_split"   # train/test will be auto-created here
MODEL_PATH = "cancer_model.h5"

# Auto split dataset if missing
if not os.path.exists(SPLIT_DIR):
    st.info("⚡ Creating train/test split automatically...")
    TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
    TEST_DIR = os.path.join(SPLIT_DIR, "test")
    for d in [TRAIN_DIR, TEST_DIR]:
        os.makedirs(os.path.join(d, "benign"), exist_ok=True)
        os.makedirs(os.path.join(d, "malignant"), exist_ok=True)

    for cls in ["benign", "malignant"]:
        imgs = [os.path.join(RAW_DATA, cls, f) for f in os.listdir(os.path.join(RAW_DATA, cls))]
        train_imgs, test_imgs = train_test_split(imgs, test_size=0.2, random_state=42)
        for f in train_imgs:
            shutil.copy(f, os.path.join(TRAIN_DIR, cls))
        for f in test_imgs:
            shutil.copy(f, os.path.join(TEST_DIR, cls))
    st.success("✅ Train/test split created!")

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

st.title("🩺 Skin Cancer Detection")
st.write("Upload a skin lesion image (will be resized to 224x224).")

# Function to train model
def train_model():
    train_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR, target_size=(224,224), batch_size=32, class_mode="binary"
    )
    test_data = test_gen.flow_from_directory(
        TEST_DIR, target_size=(224,224), batch_size=32, class_mode="binary"
    )

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    st.info("⚡ Training model... this may take a few minutes")
    model.fit(train_data, validation_data=test_data, epochs=5)
    model.save(MODEL_PATH)
    st.success("✅ Model trained and saved!")
    return model

# Load or train
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = train_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]

    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if pred < 0.5:
        st.success(f"🟢 Benign (NOT cancerous) — Confidence: {1-pred:.2f}")
    else:
        st.error(f"🔴 Malignant (CANCEROUS) — Confidence: {pred:.2f}")

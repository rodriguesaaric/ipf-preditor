# The required custom objects and utility functions must be defined here for Keras to load the model correctly.

import os
import io
import tempfile
import numpy as np
import pydicom
import cv2
import pandas as pd
from typing import Dict, Any, List

# --- TensorFlow and Keras Imports ---
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.applications import EfficientNetB3

# --- API Imports ---
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(title="FVC Prediction API", version="1.0")

# --- GLOBAL CONFIGURATION ---
# Looks for the correct .weights.h5 extension one folder up (i.e., in the root)
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'fvc_model_weights.weights.h5')
FVC_MODEL = None

# DICOM preprocessing config (MUST MATCH TRAINING)
IMG_SIZE = 256
TABULAR_FEATURES = ['Weeks', 'Age', 'FVC', 'Sex_male', 'SmokingStatus_Ex-smoker', 'SmokingStatus_Never Smoker']


# ==============================================================================
# 1. CUSTOM LOSS FUNCTION (Crucial for loading the trained model)
# ==============================================================================

def laplace_log_likelihood(y_true, y_pred):
    """
    Custom Loss Function based on the Kaggle competition (Laplace Likelihood).
    The model predicts two outputs: mu (mean/FVC) and sigma (standard deviation/Confidence).
    """
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = K.exp(log_sigma)
    y_true_fvc = y_true[:, 0]

    # Calculate negative log-likelihood for Laplace distribution
    loss = (K.abs(y_true_fvc - mu) / sigma) + K.log(2 * sigma)

    return K.mean(loss)


# ==============================================================================
# 2. MODEL DEFINITION AND LOADING
# ==============================================================================

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), tabular_dim=len(TABULAR_FEATURES)):
    """
    Defines the dual-input EfficientNet model architecture.
    """
    img_input = Input(shape=input_shape, name='image_input')
    x = Concatenate()([img_input, img_input, img_input])

    # NOTE: weights=None is used in the API build because we load them from the .h5 file.
    cnn = EfficientNetB3(weights=None, include_top=False, input_tensor=x)
    cnn.trainable = False

    image_features = cnn.output
    image_features = GlobalAveragePooling2D()(image_features)
    image_features = Dense(64, activation='relu')(image_features)

    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    tabular_features = Dense(32, activation='relu')(tabular_input)

    fused = Concatenate()([image_features, tabular_input])
    fused = Dense(64, activation='relu')(fused)

    mu_output = Dense(1, name='mu')(fused)
    sigma_output = Dense(1, name='log_sigma')(fused)

    combined_output = Concatenate(axis=-1)([mu_output, sigma_output])

    model = Model(inputs=[img_input, tabular_input], outputs=combined_output)

    model.compile(optimizer='adam', loss=laplace_log_likelihood)

    return model


def load_model():
    """
    Loads the pre-trained Keras model from the disk.
    """
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise RuntimeError("Model weights file is missing. Please ensure 'fvc_model_weights.weights.h5' is uploaded.")

    try:
        model = build_model()
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("✅ Model loaded successfully from weights.")
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load the FVC model: {e}")


# ==============================================================================
# 3. DICOM PREPROCESSING UTILITIES (MUST MATCH TRAINING)
# ==============================================================================

def preprocess_dicom(dicom_bytes: bytes) -> np.ndarray:
    """
    Reads DICOM bytes, processes the image, and returns a normalized NumPy array.
    """
    try:
        dcm_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
        img = dcm_data.pixel_array.astype(np.int16)

        # 1. Apply Rescaling and Intercept to get Hounsfield Units (HU)
        if 'RescaleSlope' in dcm_data and 'RescaleIntercept' in dcm_data:
            img = img * dcm_data.RescaleSlope + dcm_data.RescaleIntercept

        # 2. Lung Windowing and Normalization (MATCHES TRAINING)
        MIN_HU = -1000.0
        MAX_HU = -400.0

        img = np.clip(img, MIN_HU, MAX_HU)
        img = (img - MIN_HU) / (MAX_HU - MIN_HU)

        # 3. Resize and Finalize
        img = (img * 255).astype(np.uint8)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        img_final = img_resized.astype(np.float32) / 255.0
        # Add batch and channel dimension (1, IMG_SIZE, IMG_SIZE, 1)
        img_final = np.expand_dims(np.expand_dims(img_final, axis=0), axis=-1)

        return img_final

    except Exception as e:
        print(f"DICOM processing error: {e}")
        raise ValueError(f"Could not process DICOM file: {e}")


# ==============================================================================
# 4. TABULAR DATA PREPARATION (MUST MATCH TRAINING)
# ==============================================================================

def prepare_tabular_data(age: int, sex: str, smoking_status: str, weeks: int, fvc: float) -> np.ndarray:
    """
    Encodes categorical features and prepares the tabular data vector.
    """
    try:
        input_data = {
            'Weeks': [weeks],
            'Age': [age],
            'FVC': [fvc],
            'Sex': [sex],
            'SmokingStatus': [smoking_status]
        }
        df = pd.DataFrame(input_data)

        df = pd.get_dummies(df, columns=['Sex', 'SmokingStatus'], prefix=['Sex', 'SmokingStatus'])

        expected_features = TABULAR_FEATURES

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        tabular_input_vector = df[expected_features].values.astype(np.float32)

        return tabular_input_vector

    except Exception as e:
        print(f"Tabular data preparation error: {e}")
        raise ValueError(f"Could not prepare tabular data: {e}")


# ==============================================================================
# 5. API ENDPOINT DEFINITION
# ==============================================================================

@app.on_event("startup")
def startup_event():
    """Load the model when the application starts."""
    global FVC_MODEL
    try:
        FVC_MODEL = load_model()
        # Warm-up
        dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        dummy_tab = np.zeros((1, len(TABULAR_FEATURES)), dtype=np.float32)
        FVC_MODEL.predict([dummy_img, dummy_tab], verbose=0)
    except Exception as e:
        print(f"❌ API startup failed: {e}")
        FVC_MODEL = None


@app.get("/")
def read_root():
    return {"message": "FVC Progression Prediction API Status: Running", "model_loaded": FVC_MODEL is not None}


@app.post("/predict")
async def predict_fvc(
        ctScan: UploadFile = File(..., description="DICOM CT Scan File (.dcm)"),
        weeks: int = Form(..., description="Target Weeks (Relative to CT)"),
        fvc: float = Form(..., description="Recorded FVC (Baseline) in mL"),
        age: int = Form(..., description="Age in Years"),
        sex: str = Form(..., description="Sex: male or female"),
        smokingStatus: str = Form(..., description="Smoking Status: Never Smoker, Ex-smoker, or Current Smoker")
) -> Dict[str, Any]:
    """
    Receives patient data and CT file, runs model inference, and returns predicted FVC and Confidence.
    """
    if FVC_MODEL is None:
        raise HTTPException(status_code=503,
                            detail="Model is not yet loaded or failed to load on startup. Check server logs.")

    # 1. Read and Process DICOM file
    try:
        dicom_bytes = await ctScan.read()
        image_input = preprocess_dicom(dicom_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image file error: {e}")

    # 2. Prepare Tabular Data
    try:
        tabular_input = prepare_tabular_data(
            age=age,
            sex=sex,
            smoking_status=smokingStatus,
            weeks=weeks,
            fvc=fvc
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tabular data error: {e}")

    # 3. Model Inference
    try:
        prediction_output = FVC_MODEL.predict([image_input, tabular_input], verbose=0)
        mu_pred = prediction_output[0, 0]
        log_sigma_pred = prediction_output[0, 1]

        confidence_ml = np.exp(log_sigma_pred)

    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # 4. Return Structured Output
    return {
        "status": "success",
        "FVC": round(float(mu_pred), 2),
        "Confidence": round(float(confidence_ml), 2),
        "Patient_Week_Stub": f"P_WK_{weeks}"
    }
# api/predict.py
# Reworked and hardened version of your predict API.
# Fixes: proper concatenation of tabular features, robust model path handling,
# clearer startup errors, and safer runtime checks.

import os
import io
import numpy as np
import pydicom
import cv2
import pandas as pd
from typing import Dict, Any

# TensorFlow / Keras imports
# NOTE: Installing TensorFlow in Vercel may be problematic (size/memory). See README notes.
try:
    # reduce TF logging
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate
    from tensorflow.keras.applications import EfficientNetB3
except Exception as e:
    # We still import FastAPI below so the app can start and show a clear error message.
    keras = None
    TF_IMPORT_ERROR = e

from fastapi import FastAPI, File, UploadFile, Form, HTTPException

app = FastAPI(title="FVC Prediction API", version="1.0")

# -----------------------------------------------------------------------------
# Config - adjust as needed
# -----------------------------------------------------------------------------
IMG_SIZE = 256
TABULAR_FEATURES = [
    "Weeks",
    "Age",
    "FVC",
    "Sex_male",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never Smoker"
]

# Model weights location:
# - You can put the weights file at repository root named exactly
#   "fvc_model_weights.weights.h5" or set MODEL_WEIGHTS_PATH env var to override.
DEFAULT_WEIGHTS_FILENAME = "fvc_model_weights.weights.h5"
MODEL_WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS_PATH") or os.path.join(
    os.path.dirname(__file__), "..", DEFAULT_WEIGHTS_FILENAME
)
MODEL_WEIGHTS_PATH = os.path.abspath(MODEL_WEIGHTS_PATH)

FVC_MODEL = None


# ---------------------------------------------------------------------------
# Custom loss (Laplace log-likelihood) - needed if weights were saved from a
# model that used this custom loss during compile.
# ---------------------------------------------------------------------------
def laplace_log_likelihood(y_true, y_pred):
    mu = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma = K.exp(log_sigma)
    y_true_fvc = y_true[:, 0]
    loss = (K.abs(y_true_fvc - mu) / sigma) + K.log(2.0 * sigma)
    return K.mean(loss)


# ---------------------------------------------------------------------------
# Build model (must match the architecture used during training)
# ---------------------------------------------------------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), tabular_dim=len(TABULAR_FEATURES)):
    img_input = Input(shape=input_shape, name="image_input")
    # Convert single channel -> 3 channels by concatenation (matches your training pipeline)
    x = Concatenate()([img_input, img_input, img_input])

    # Use weights=None because we will load weights from the provided file
    cnn = EfficientNetB3(weights=None, include_top=False, input_tensor=x)
    cnn.trainable = False

    image_features = GlobalAveragePooling2D()(cnn.output)
    image_features = Dense(64, activation="relu")(image_features)

    tabular_input = Input(shape=(tabular_dim,), name="tabular_input")
    # Use the computed tabular features (not the raw tabular_input) when fusing
    tabular_features = Dense(32, activation="relu")(tabular_input)

    fused = Concatenate()([image_features, tabular_features])
    fused = Dense(64, activation="relu")(fused)

    mu_output = Dense(1, name="mu")(fused)
    log_sigma_output = Dense(1, name="log_sigma")(fused)

    combined_output = Concatenate(axis=-1)([mu_output, log_sigma_output])

    model = Model(inputs=[img_input, tabular_input], outputs=combined_output)
    model.compile(optimizer="adam", loss=laplace_log_likelihood)
    return model


def find_weights_candidate():
    """
    Try a few candidate locations for the weights file (helpful for local vs server env).
    """
    candidates = [
        MODEL_WEIGHTS_PATH,
        os.path.join(os.path.dirname(__file__), DEFAULT_WEIGHTS_FILENAME),
        os.path.join(os.path.dirname(__file__), "..", DEFAULT_WEIGHTS_FILENAME),
        os.path.join(os.getcwd(), DEFAULT_WEIGHTS_FILENAME)
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def load_model_robust():
    global keras
    if keras is None:
        raise RuntimeError(f"TensorFlow import failed: {TF_IMPORT_ERROR}")

    weights_path = find_weights_candidate()
    if not weights_path:
        raise RuntimeError(
            "Model weights file not found. Expected one of the candidate paths. "
            f"Set MODEL_WEIGHTS_PATH environment variable if needed. Searched: {MODEL_WEIGHTS_PATH}"
        )
    try:
        model = build_model()
        model.load_weights(weights_path)
        print(f"✅ Model loaded from: {weights_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from '{weights_path}': {e}")


# ---------------------------------------------------------------------------
# DICOM preprocessing
# ---------------------------------------------------------------------------
def preprocess_dicom(dicom_bytes: bytes) -> np.ndarray:
    try:
        dcm_data = pydicom.dcmread(io.BytesIO(dicom_bytes))
        img = dcm_data.pixel_array.astype(np.int16)

        if hasattr(dcm_data, "RescaleSlope") and hasattr(dcm_data, "RescaleIntercept"):
            img = img * float(dcm_data.RescaleSlope) + float(dcm_data.RescaleIntercept)

        MIN_HU = -1000.0
        MAX_HU = -400.0
        img = np.clip(img, MIN_HU, MAX_HU)
        img = (img - MIN_HU) / (MAX_HU - MIN_HU)
        img = (img * 255.0).astype(np.uint8)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img_final = img_resized.astype(np.float32) / 255.0
        img_final = np.expand_dims(np.expand_dims(img_final, axis=0), axis=-1)  # (1, H, W, 1)
        return img_final
    except Exception as e:
        raise ValueError(f"Could not process DICOM: {e}")


# ---------------------------------------------------------------------------
# Tabular data preparation
# ---------------------------------------------------------------------------
def prepare_tabular_data(age: int, sex: str, smoking_status: str, weeks: int, fvc: float) -> np.ndarray:
    try:
        input_data = {
            "Weeks": [weeks],
            "Age": [age],
            "FVC": [fvc],
            "Sex": [sex],
            "SmokingStatus": [smoking_status],
        }
        df = pd.DataFrame(input_data)
        df = pd.get_dummies(df, columns=["Sex", "SmokingStatus"], prefix=["Sex", "SmokingStatus"])

        for col in TABULAR_FEATURES:
            if col not in df.columns:
                df[col] = 0

        tab_vec = df[TABULAR_FEATURES].values.astype(np.float32)
        return tab_vec
    except Exception as e:
        raise ValueError(f"Could not prepare tabular data: {e}")


# ---------------------------------------------------------------------------
# Startup: try loading model (but handle failures gracefully)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global FVC_MODEL
    try:
        FVC_MODEL = load_model_robust()
        # Warmup predict to avoid extreme cold start delay later
        dummy_img = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        dummy_tab = np.zeros((1, len(TABULAR_FEATURES)), dtype=np.float32)
        FVC_MODEL.predict([dummy_img, dummy_tab], verbose=0)
    except Exception as e:
        # Don't crash the server — but record model as None and print reason.
        print(f"❌ API startup: could not load model: {e}")
        FVC_MODEL = None


@app.get("/")
def read_root():
    return {"message": "FVC Prediction API", "model_loaded": FVC_MODEL is not None}


@app.post("/predict")
async def predict_fvc(
    ctScan: UploadFile = File(..., description="DICOM CT Scan File (.dcm)"),
    weeks: int = Form(...),
    fvc: float = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    smokingStatus: str = Form(...)
) -> Dict[str, Any]:
    if FVC_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # 1. DICOM -> image tensor
    try:
        dicom_bytes = await ctScan.read()
        image_input = preprocess_dicom(dicom_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    # 2. Tabular
    try:
        tabular_input = prepare_tabular_data(age=age, sex=sex, smoking_status=smokingStatus, weeks=weeks, fvc=fvc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tabular processing error: {e}")

    # 3. Predict
    try:
        pred = FVC_MODEL.predict([image_input, tabular_input], verbose=0)
        mu = float(pred[0, 0])
        log_sigma = float(pred[0, 1])
        confidence = float(np.exp(log_sigma))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "status": "success",
        "FVC": round(mu, 2),
        "Confidence": round(confidence, 2),
        "Patient_Week_Stub": f"P_WK_{weeks}"
    }

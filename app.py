from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sqlite3

# -----------------------------
# CONFIG
# -----------------------------
USE_REAL_MODEL = False  # Change to True when model.h5 is ready

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODEL (only if real)
# -----------------------------
if USE_REAL_MODEL:
    import tensorflow as tf
    model = tf.keras.models.load_model("model/model.h5")

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            disease TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image_path):
    if USE_REAL_MODEL:
        # Real prediction logic here
        prediction = model.predict(image_path)
        return {
            "disease": "Real Disease",
            "confidence": 0.95
        }
    else:
        # Dummy prediction (for now)
        return {
            "disease": "Tomato Early Blight",
            "confidence": 0.91
        }

# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Crop Disease Detection Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Get prediction
    result = predict_image(file_path)

    # Save to database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (image_name, disease, confidence)
        VALUES (?, ?, ?)
    """, (file.filename, result["disease"], result["confidence"]))
    conn.commit()
    conn.close()

    return result

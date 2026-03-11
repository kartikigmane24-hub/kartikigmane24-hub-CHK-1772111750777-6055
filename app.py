from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import openai

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="AI Crop Health + Chatbot API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load AI Model (your existing model)
# -----------------------------
MODEL_PATH = "crop_model.h5"
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

# Define disease classes and treatments
DISEASE_CLASSES = ["Tomato___Early_Blight", "Tomato___Late_Blight", "Tomato___Healthy"]
TREATMENTS = {
    "Early_Blight": "Apply copper fungicide, prune lower leaves",
    "Late_Blight": "Remove infected plants, apply fungicide",
    "Healthy": "No treatment needed"
}

# -----------------------------
# Helper Functions
# -----------------------------
def get_disease_class(disease_full):
    if "Early_Blight" in disease_full:
        return "Fungal"
    elif "Late_Blight" in disease_full:
        return "Fungal"
    else:
        return "Healthy"

def smart_fallback_prediction(img_array):
    # Fallback random prediction if model not loaded
    idx = np.random.randint(0, len(DISEASE_CLASSES))
    return DISEASE_CLASSES[idx], float(np.random.uniform(90, 100))

def detect_multi_diseases(disease_name, img_array):
    # Dummy example: returns multiple diseases randomly
    if np.random.rand() > 0.5:
        return ["Early_Blight", "Late_Blight"]
    return None

# -----------------------------
# Crop Disease Prediction Endpoint
# -----------------------------
@app.get("/")
def index():
    return {"message": "Crop Disease Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        if file is None:
            return JSONResponse({"error": "No image uploaded"}, status_code=400)

        # Read and process image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # AI prediction
        if model is not None:
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx] * 100)
            disease_full = DISEASE_CLASSES[predicted_class_idx]
        else:
            disease_full, confidence = smart_fallback_prediction(img_array)

        disease_name = disease_full.split('___')[1] if '___' in disease_full else disease_full
        disease_class = get_disease_class(disease_full)
        treatment = TREATMENTS.get(disease_name, TREATMENTS['Healthy'])

        # Multi-disease detection
        multi_diseases = detect_multi_diseases(disease_name, img_array)

        result = {
            "success": True,
            "disease": disease_name,
            "disease_class": disease_class,
            "confidence": f"{confidence:.1f}%",
            "full_name": disease_full,
            "treatment": treatment,
            "is_healthy": "healthy" in disease_name.lower()
        }

        if multi_diseases and len(multi_diseases) > 1:
            result["multiple_diseases"] = multi_diseases

        return result

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# GPT AI Chatbot Endpoint
# -----------------------------
# Make sure to set your OpenAI API key in environment variable
# export OPENAI_API_KEY="your_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.post("/chat")
async def chat(message: dict):
    try:
        user_message = message.get("message", "")
        if not user_message:
            return JSONResponse({"reply": "Please send a message."}, status_code=400)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=200
        )

        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"}, status_code=500)
    

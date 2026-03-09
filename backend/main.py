from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = FastAPI()

# Load your trained model
model = load_model('model.h5')

# Classes must match training
diseases = ['Potato___Early_blight', 'Tomato_Early_Blight', 'Tomato___Early_blight', 'Tomato___healthy']

# Helper function to preprocess image
def prepare_image(file):
    img = image.load_img(file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# /predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_array = prepare_image(file.file)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))
    return {"disease": diseases[class_idx], "confidence": confidence}

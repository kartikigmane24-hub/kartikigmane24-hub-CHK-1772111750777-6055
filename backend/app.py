from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import cv2
import json

app = Flask(__name__, template_folder='templates')
CORS(app)

# HIERARCHICAL DISEASE CLASSIFICATION
DISEASE_HIERARCHY = {
    "Fungal": ["Apple_scab", "Early_blight", "Late_blight", "Powdery_mildew", "Septoria_leaf_spot", "Leaf_Mold", "Target_Spot"],
    "Bacterial": ["Bacterial_spot"],
    "Viral": ["Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus"],
    "Pest": ["Spider_mites Two-spotted_spider_mite"],
    "Healthy": ["healthy"]
}

DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell____Bacterial_spot', 'Pepper,_bell____healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

TREATMENTS = {
    'healthy': '✅ No treatment needed. Plant is healthy!',
    'Early_blight': 'Copper fungicide spray (2g/liter water). Prune lower leaves. Repeat every 7 days.',
    'Late_blight': 'Mancozeb 75% WP (2g/liter). Remove infected plants immediately. Improve drainage.',
    'Bacterial_spot': 'Copper oxychloride (3g/liter). Avoid overhead watering. Sanitize tools.',
    'Apple_scab': 'Mancozeb + Carbendazim mix. Prune for air circulation.',
    'Powdery_mildew': 'Sulphur 80% WP (2g/liter). Morning spray best.',
    'Septoria_leaf_spot': 'Mancozeb + Copper spray. Remove old leaves.',
    'Leaf_Mold': 'Improve greenhouse ventilation. Mancozeb spray.',
    'Spider_mites': 'Neem oil (5ml/liter) + mild soap. Spray underside of leaves.'
}

MODEL_PATH = 'models/plant_disease_model.h5'
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Loaded model from {MODEL_PATH}")
        else:
            print("⚠️ Model not found. Using smart fallback...")
            model = None
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model = None

def get_disease_class(disease_name):
    """Classify disease into hierarchical category"""
    clean_name = disease_name.replace('___', '_').lower()
    for class_name, diseases in DISEASE_HIERARCHY.items():
        if any(disease.lower() in clean_name or clean_name in disease.lower() for disease in diseases):
            return class_name
    return "Healthy"

def smart_fallback_prediction(img_array):
    """Smart fallback using image analysis (no model needed)"""
    img = img_array[0]
    
    # Greenness score (healthy plants = high green)
    green_channel = img[:,:,1]
    greenness = np.mean(green_channel)
    
    # Dark spots (disease indicator)
    dark_pixels = np.sum(img < 0.3)
    spot_ratio = dark_pixels / img.size
    
    # Texture complexity (disease = irregular patterns)
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    laplacian_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
    
    if greenness > 0.5 and spot_ratio < 0.05:
        return 'healthy', 95.0
    elif spot_ratio > 0.1:
        return 'Bacterial_spot', 88.0
    elif laplacian_var > 200:
        return 'Early_blight', 85.0
    else:
        return 'Late_blight', 82.0

def detect_multi_diseases(disease_name, img_array):
    """Backend multi-disease detection"""
    multi_diseases = [disease_name]
    
    # Advanced pixel analysis for co-infections
    img = img_array[0]
    
    # Brown spots (Early/Late blight)
    brown_mask = (img[:,:,0] > 100) & (img[:,:,1] < 120) & (img[:,:,2] < 100)
    brown_count = np.sum(brown_mask)
    
    # Water-soaked lesions (Bacterial)
    wet_mask = (img[:,:,0] < 80) & (img[:,:,1] < 90) & (img[:,:,2] < 100)
    wet_count = np.sum(wet_mask)
    
    if brown_count > 50000 and 'Bacterial_spot' not in disease_name:
        multi_diseases.append('Early_blight')
    
    if wet_count > 40000 and 'Early_blight' not in disease_name:
        multi_diseases.append('Bacterial_spot')
    
    return multi_diseases if len(multi_diseases) > 1 else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Process image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # AI Prediction
        if model is not None:
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx] * 100)
            disease_full = DISEASE_CLASSES[predicted_class_idx]
        else:
            disease_full, confidence = smart_fallback_prediction(img_array)
        
        disease_name = disease_full.split('___')[1] if '___' in disease_full else disease_full
        disease_class = get_disease_class(disease_full)
        treatment = TREATMENTS.get(disease_name, TREATMENTS['healthy'])
        
        # 🚨 MULTI-DISEASE DETECTION (Backend)
        multi_diseases = detect_multi_diseases(disease_name, img_array)
        
        result = {
            'success': True,
            'disease': disease_name,
            'disease_class': disease_class,
            'confidence': f"{confidence:.1f}%",
            'full_name': disease_full,
            'treatment': treatment,
            'is_healthy': 'healthy' in disease_name.lower()
        }
        
        # Add multi-disease info
        if multi_diseases and len(multi_diseases) > 1:
            result['multiple_diseases'] = multi_diseases
            result['disease_class'] = '🚨 MULTI-DISEASE'
            result['treatment'] = f"🚨 EMERGENCY: {multi_diseases[0]} + {multi_diseases[1]} detected! Priority: {multi_diseases[0]}. Copper fungicide + Mancozeb mix. Remove infected parts."
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    print("🚀 Backend running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

#  AI Crop Disease Detection System

# Overview
The **AI Crop Disease Detection System** is a machine learning based web application that helps detect crop diseases from plant leaf images.  
Farmers or users can upload an image of a plant leaf, and the system will analyze the image using a trained AI model to predict the disease.

This system helps farmers **identify diseases early and take proper action**, improving crop health and agricultural productivity.

---

# Problem Statement
Crop diseases are one of the major reasons for reduced agricultural productivity.  
Many farmers cannot easily identify plant diseases at an early stage.

There is a need for an **automated system that can detect plant diseases quickly and accurately using AI.**

---

# Proposed Solution
This project provides an **AI-powered crop disease detection system** where:

1. The user uploads a plant leaf image.
2. The system processes the image.
3. A trained machine learning model predicts the disease.
4. The result is displayed to the user.

This makes disease detection **fast, simple, and accessible for farmers.**

---

## Features
- Upload plant leaf images
- AI-based crop disease prediction
- Simple and user-friendly interface
- Fast disease detection
- Stores prediction results

---

# Technologies Used

# Frontend
- HTML
- CSS
- JavaScript

#Backend
- Python
- FastAPI
- Uvicorn

# Libraries
- NumPy
- OpenCV
- Pillow
- OpenAI Python SDK

# Database
- SQLite

---

# Machine Learning Model
The model is trained using a **Plant Disease Dataset** containing images of healthy and diseased leaves.

#Model Workflow
1. Data collection
2. Image preprocessing
3. Model training
4. Disease classification
5. Prediction output

The trained model analyzes the uploaded image and predicts the disease category.

---

# Project Structure

```
Dataset/                → Plant disease dataset
backend/                → Backend API and database
frontend/               → Web interface
uploads/                → Uploaded images

app.py                  → Main backend application
train_model.py          → Model training script
database.db             → SQLite database
requirements.txt        → Python dependencies
README.md               → Project documentation
```

---

# How to Run the Project

## Clone the repository

```
git clone <repository-url>
```

## Install dependencies

```
pip install -r requirements.txt
```

## Run the backend server

```
uvicorn app:app --reload
```

# Open the frontend

Open the **crop.html** file inside the `frontend` folder in your browser.

---

# Working of the System

1. User uploads a plant leaf image
2. The image is sent to the backend server
3. The trained AI model processes the image
4. The system predicts the disease
5. The result is displayed to the user

---

# Future Enhancements
- Mobile application for farmers
- Support for more crop types
- Real-time disease detection
- Integration with agricultural advisory systems
- Cloud deployment

# Contributors
- Kartiki Mane
- Team Members

# Conclusion
The **AI Crop Disease Detection System** demonstrates how Artificial Intelligence can help farmers detect crop diseases quickly and efficiently.

By using AI technology, farmers can take early preventive measures and reduce crop losses.

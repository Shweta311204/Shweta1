
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
from PIL import Image
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
run_with_ngrok(app)  # Starts ngrok when you run the app

# ========== Dummy ML Functions ==========

def predict_skin_disease(image):
    # Dummy function, replace with your actual model prediction
    return "Melanoma", 0.93

def predict_voice_health(audio_data):
    # Dummy function, replace with your actual model prediction
    return "Likely Parkinson's", 0.89

def predict_cancer_cells(image):
    # Dummy function, replace with your actual model prediction
    return "Malignant Tissue", 0.91

# ========== Routes ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file)
    result, confidence = predict_skin_disease(image)
    return jsonify({'prediction': result, 'confidence': f"{confidence*100:.2f}%"})

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    audio_bytes = file.read()
    audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio_np, sr=sr)
    plt.close(fig)
    
    result, confidence = predict_voice_health(audio_np)
    return jsonify({'prediction': result, 'confidence': f"{confidence*100:.2f}%"})

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    image = Image.open(file)
    result, confidence = predict_cancer_cells(image)
    return jsonify({'prediction': result, 'confidence': f"{confidence*100:.2f}%"})



if __name__ == '__main__':
    app.run()

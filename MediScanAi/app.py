
from flask import Flask, render_template, request
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
    return "Detected Condition: Melanoma", 0.93

def predict_voice_health(audio_data):
    # Dummy function, replace with your actual model prediction
    return "Prediction: Likely Parkinson's Disease", 0.89

def predict_cancer_cells(image):
    # Dummy function, replace with your actual model prediction
    return "Diagnosis: Malignant Tissue", 0.91

# ========== Routes ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_skin', methods=['POST'])
def predict_skin():
    if 'image' not in request.files:
        return render_template('result.html', prediction="No image uploaded", confidence="N/A")

    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', prediction="No image selected", confidence="N/A")

    image = Image.open(file)
    result, confidence = predict_skin_disease(image)
    return render_template('result.html', prediction=result, confidence=f"{confidence*100:.2f}%")

@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    if 'audio' not in request.files:
        return render_template('result.html', prediction="No audio uploaded", confidence="N/A")

    file = request.files['audio']
    if file.filename == '':
        return render_template('result.html', prediction="No audio selected", confidence="N/A")

    audio_bytes = file.read()
    audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # Optional: Generate waveform image (hidden from final render)
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio_np, sr=sr)
    plt.close(fig)

    result, confidence = predict_voice_health(audio_np)
    return render_template('result.html', prediction=result, confidence=f"{confidence*100:.2f}%")

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    if 'image' not in request.files:
        return render_template('result.html', prediction="No image uploaded", confidence="N/A")

    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', prediction="No image selected", confidence="N/A")

    image = Image.open(file)
    result, confidence = predict_cancer_cells(image)
    return render_template('result.html', prediction=result, confidence=f"{confidence*100:.2f}%")

if __name__ == '__main__':
    app.run()

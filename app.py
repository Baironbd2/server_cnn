from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
from src.Utils import (
    detect_faces,
    draw_box,
    load_image,
    detect_emotion,
    detect_emotion_array,
    predecir,
    sum_emotions,
    convert_to_percentage,
    process_images_from_directory,
    process_emotion_from_directory
)

import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DOWNLOAD_FOLDER = "download"
modelo = joblib.load('./models/modelo.pkl')

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER

CORS(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

parcial_value = None
emotion_arrays = []
emotions_parciales  = []

@app.route("/upload", methods=["POST"])
def upload_file():
    global emotion_arrays, emotions_parciales
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        process_images_from_directory(app.config["UPLOAD_FOLDER"], app.config["DOWNLOAD_FOLDER"])
        emotion_array = process_emotion_from_directory(app.config["DOWNLOAD_FOLDER"])
        summed_emotions = sum_emotions(emotion_array)
        emotions_parciales = convert_to_percentage(summed_emotions)
        print(f'Emociones parciales: {emotions_parciales}')     
    return jsonify({"success": "File uploaded successfully"}),200
           
@app.route("/parcial", methods=["POST"])
def pre():
    global parcial_value
    data = request.get_json()
    parcial = float(data['parcial'])
    parcial_value = parcial
    print(parcial_value)   
    return jsonify({"success": f'Parcial {parcial_value} received, waiting for images'}), 200

@app.route("/prediccion", methods=["POST"])
def predecir():
    global parcial_value, emotions_parciales
    if len(emotions_parciales) == 0:
        return jsonify({"error": "No emotions data available"}), 400
    
    data = [
        parcial_value,
        emotions_parciales[1],
        emotions_parciales[5], 
        emotions_parciales[3],  
        emotions_parciales[4],  
        emotions_parciales[0], 
        emotions_parciales[2]  
    ]
    print(data)
    prediction = modelo.predict([data])[0]
    print(prediction)
    return jsonify({"prediccion": prediction})

@app.route("/reset", methods=["POST"])
def reset():
    global emotion_arrays,emotions_parciales,parcial_value
    emotion_arrays = []
    emotions_parciales = []
    parcial_value = None
    for folder in [UPLOAD_FOLDER, DOWNLOAD_FOLDER]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    return jsonify({"success": "Reset completed"}), 200

@app.route('/processed/<filename>', methods=['GET'])
def get_processed_image(filename):
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)

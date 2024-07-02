import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#Lista de emociones
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#cargar modelos
modelemotion = load_model('./models/Emotions6.h5')
modelo = joblib.load('./models/modelo.pkl')
with tf.io.gfile.GFile('./models/graph_face.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as mobilenet:
    tf.import_graph_def(graph_def, name='')

#Detectar rostros
def detect_faces(image, score_threshold=0.7):
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image, axis=0)
    with tf.compat.v1.Session(graph=mobilenet) as sess:
        image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
        boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
        scores = mobilenet.get_tensor_by_name('detection_scores:0')
        (boxes, scores) = sess.run([boxes, scores], feed_dict={image_tensor: img})
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    idx = np.where(scores >= score_threshold)[0]
    bboxes = []

    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index, :]
        (left, right, top, bottom) = (xmin * imw, xmax * imw, ymin * imh, ymax * imh)
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        bboxes.append([left, right, top, bottom])
    return bboxes

#Dibujar recuadros
def draw_box(image, box, label,color, line_width=6):
    if box == []:
        return image
    else:
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), color, line_width)
        cv2.putText(image, label, (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image

#Cargar imagen
def load_image(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

#Detectar emociones
def detect_emotion(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    emotion_prediction = modelemotion.predict(face)
    emotion_label = class_names[emotion_prediction.argmax(axis=1)[0]]
    return emotion_label

#Array de emociones
def detect_emotion_array(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    emotion_prediction = modelemotion.predict(face)[0]
    return emotion_prediction

#Suma del array de emociones
def sum_emotions(emotion_arrays):
    emotion_arrays_np = np.array(emotion_arrays)
    summed_emotions = np.sum(emotion_arrays_np, axis=0)
    np.set_printoptions(suppress=True, precision=8)
    return summed_emotions

#Emociones en porcentaje
def convert_to_percentage(array):
    total = np.sum(array)
    percentage_array = np.round((array / total) * 100, 2)
    return percentage_array

#Guardar imagenes con rostros detectados
def process_images_from_directory(directory, output_directory):
    filenames = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    if len(filenames) >= 3:
        for name in filenames:
            filepath = os.path.join(directory, name)
            image = load_image(filepath)
            bboxes = detect_faces(image)
            for box in bboxes:
                face = image[box[2]:box[3], box[0]:box[1]]
                emotion = detect_emotion(face)
                image = draw_box(image, box, emotion, (0, 255, 0))
            output_filepath = os.path.join(output_directory, name)
            cv2.imwrite(output_filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Imagen procesada y guardada en {output_filepath}")

#Guardar las emocines
def process_emotion_from_directory(directory):   
    emotion_array=[]
    filenames = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    if len(filenames) >= 3:
        for name in filenames:
            filepath = os.path.join(directory, name)
            image = load_image(filepath)
            bboxes = detect_faces(image)
            for box in bboxes:
                face = image[box[2]:box[3], box[0]:box[1]]
                array = detect_emotion_array(face)
                emotion_array.append(array)
    return(emotion_array)

def predecir (data):
    prediction = modelo.predict(data)
    return prediction

# process_images_from_directory(UPLOAD_DIR, DOWNLOAD_DIR)

# emotion_arrays = process_emotion_from_directory(DOWNLOAD_DIR)
# summed_emotions = sum_emotions(emotion_arrays)
# data = convert_to_percentage(summed_emotions)
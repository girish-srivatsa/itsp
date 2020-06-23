import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def facecrop(image):
    facedata = r"/home/prajeeth/Environments/tf_env/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image, 0)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        fname, ext = os.path.splitext(image)
        new_path = fname + "_cropped_" + ext
        cv2.imwrite(new_path, sub_face)

    return new_path


Emodel = load_model('Emotion-Model.h5')

def emotion_function(img_path):
    face_path = facecrop(img_path)
    face = Image.open(face_path)
    face = face.resize((48, 48))
    face_arr = np.asarray(face)
    face_arr = face_arr / 255
    face_arr.shape = (1, 48, 48, 1)

    y_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    pred_arr = Emodel.predict(face_arr, verbose = 1)

    return y_classes[np.argmax(pred_arr)]

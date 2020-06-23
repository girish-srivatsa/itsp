import dlib
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames


Emodel = load_model('Emotion-Model.h5')
    
def emotion_function(image_path):

    image = np.asarray(Image.open(image_path))
    detected_faces = detect_faces(image)

    # Crop faces and plot
    for n, face_rect in enumerate(detected_faces):
        face = Image.fromarray(image).crop(face_rect)
        face.thumbnail((48, 48), Image.ANTIALIAS)
        face = face.convert('L')
       
    face_arr = np.asarray(face)
    face_arr = face_arr / 255
    face_arr.shape = (1, 48, 48, 1)

    y_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    pred_arr = Emodel.predict(face_arr, verbose = 1)
    print(y_classes[np.argmax(pred_arr)])
    return y_classes[np.argmax(pred_arr)]
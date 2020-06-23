import os
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensor
import keras.backend.tensorflow_backend as tb

# For running in a windows environment
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tb._SYMBOLIC_SCOPE.value = True

# Define a flask app
app = Flask(__name__)

y_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("File path: ", file_path)
        return tensor.emotion_function(file_path)
    return None


if __name__ == '__main__':
    app.run(threaded=False, host="0.0.0.0", port=8082)

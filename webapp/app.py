import os
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensor1
import keras.backend.tensorflow_backend as tb

tb._SYMBOLIC_SCOPE.value = True

# Define a flask app
app = Flask(__name__)

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
        return tensor1.emotion_function(file_path)
    return None


if __name__ == '__main__':
    app.run(threaded=False)

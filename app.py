from flask import Flask, request, render_template, redirect, url_for
from cnnClassifier.pipeline.prediction import PredictPipeline
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def home():
    return redirect(url_for('main'))

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        pipeline = PredictPipeline(filepath)
        prediction = pipeline.predict()
        return render_template('result.html', prediction=prediction[0]['image'], filename=file.filename)
    return None

if __name__ == "__main__":
    app.run(debug=True)

    app.run(host="0.0.0.0" ,port=5000)
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('shoppers_model_202211938.pkl')

# Set this to match your model input size
num_features = 25

@app.route('/')
def form():
    return render_template('form.html', num_features=num_features)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])[0]
    return f"<h1>Prediction: {'Purchase' if prediction else 'No Purchase'}</h1>"

if __name__ == '__main__':
    app.run(debug=True)

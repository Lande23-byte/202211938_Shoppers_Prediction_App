from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model_202211938.pkl")
NUM_FEATURES = 17  # Replace with actual number of features in X

@app.route('/')
def home():
    return render_template('form.html', NUM_FEATURES=NUM_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        prediction = model.predict([data])[0]
        result = '✅ Will Purchase' if prediction == 1 else '❌ Will Not Purchase'
    except:
        result = "⚠️ Invalid input. Please enter numeric values only."

    return render_template('form.html', prediction=result, NUM_FEATURES=NUM_FEATURES)

if __name__ == '__main__':
    app.run(debug=True)

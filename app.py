from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = None  # Global variable to store the model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global model
    file = request.files['file']
    if not file:
        return "No file"

    # Load dataset
    df = pd.read_csv(file)
    # Assume 'Had Heart Attack' is the label for simplicity
    X = df.drop('Had Heart Attack', axis=1)
    y = df['Had Heart Attack']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return 'Model trained successfully!'

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if not model:
        return jsonify({'error': 'Model is not trained yet.'})

    data = request.get_json()
    # Assuming data is a dictionary mapping feature names to values
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

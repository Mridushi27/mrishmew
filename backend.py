from flask import Flask, request, jsonify
import sqlite3
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
try:
    with open('cancer_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'cancer_prediction_model.pkl' not found. Please train the model first.")
    exit(1)

# Database setup
def init_db():
    conn = sqlite3.connect('cancer_prediction.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            age INTEGER,
            blood_test_data TEXT,
            prediction_result TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debug statement

        # Ensure blood_test_data is provided and is a list
        blood_test_data = data.get('blood_test_data')
        if not blood_test_data or not isinstance(blood_test_data, list):
            return jsonify({'error': 'Invalid blood_test_data. Must be a list of numbers.'}), 400

        # Make prediction
        prediction = model.predict([blood_test_data])
        result = "Cancer Detected" if prediction[0] == 1 else "No Cancer Detected"
        print("Prediction result:", result)  # Debug statement

        # Save to database
        conn = sqlite3.connect('cancer_prediction.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO test_results (user_name, age, blood_test_data, prediction_result)
            VALUES (?, ?, ?, ?)
        ''', (data.get('user_name'), data.get('age'), str(blood_test_data), result))
        conn.commit()
        conn.close()

        return jsonify({'prediction': result})
    except Exception as e:
        print("Error:", str(e))  # Debug statement
        return jsonify({'error': str(e)}), 500

# Root endpoint
@app.route('/')
def home():
    return "Cancer Prediction Backend is running!"

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

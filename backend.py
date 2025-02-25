from flask import Flask, request, jsonify
import sqlite3
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
with open('cancer_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

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
    data = request.json
    blood_test_data = data['blood_test_data']  # Example: [0.5, 0.3, 0.2, ...]
    prediction = model.predict([blood_test_data])
    result = "Cancer Detected" if prediction[0] == 1 else "No Cancer Detected"

    # Save to database
    conn = sqlite3.connect('cancer_prediction.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO test_results (user_name, age, blood_test_data, prediction_result)
        VALUES (?, ?, ?, ?)
    ''', (data['user_name'], data['age'], str(blood_test_data), result))
    conn.commit()
    conn.close()

    return jsonify({'prediction': result})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

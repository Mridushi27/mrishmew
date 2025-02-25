from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Load dataset (replace with your dataset)
data = pd.read_csv('cancer_blood_test_data.csv')
X = data.drop('target', axis=1)  # Features
y = data['target']  # Labels (1: Cancer, 0: No Cancer)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('cancer_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as cancer_prediction_model.pkl")

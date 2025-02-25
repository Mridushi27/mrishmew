from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# Example dataset (replace with your actual dataset)
data = {
    'feature1': [0.5, 0.3, 0.2, 0.8, 0.7],
    'feature2': [0.1, 0.4, 0.3, 0.9, 0.5],
    'feature3': [0.2, 0.5, 0.1, 0.7, 0.4],
    'target': [1, 0, 0, 1, 1]  # 1: Cancer, 0: No Cancer
}
df = pd.DataFrame(data)

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('cancer_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as cancer_prediction_model.pkl")

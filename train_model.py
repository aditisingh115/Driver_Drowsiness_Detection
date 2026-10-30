import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset"
MODEL_PATH = "models/eye_state_model.pkl"

X = []
y = []

for label, folder in enumerate(["close", "open"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (32, 32))
        image = image.astype("float32") / 255.0

        X.append(image.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Total images:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", class_weight="balanced")
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model training completed!")
print("Model saved at:", MODEL_PATH)
import matplotlib.pyplot as plt

for folder in ["close", "open"]:
    folder_path = os.path.join(DATASET_PATH, folder)
    filename = os.listdir(folder_path)[0]
    image_path = os.path.join(folder_path, filename)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (32, 32))
    features = image.astype("float32").flatten() / 255.0

    prediction = int(model.predict([features])[0])

    print(folder, "image prediction:", prediction)
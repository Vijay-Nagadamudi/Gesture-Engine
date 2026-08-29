import joblib
import numpy as np
from collections import deque
import warnings


MODEL_PATH = "models/trained/svm_model.joblib"

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 7


class GesturePredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        self.prediction_history = deque(
            maxlen=SMOOTHING_WINDOW
        )

    def predict(self, features):
        features = np.array(features).reshape(1, -1)

        probabilities = self.model.predict_proba(features)

        best_index = np.argmax(probabilities)

        confidence = probabilities[0][best_index]

        prediction = self.model.classes_[best_index]

        if confidence < CONFIDENCE_THRESHOLD:
            self.prediction_history.clear()
            return "UNKNOWN", confidence

        self.prediction_history.append(prediction)

        stable_prediction = max(
            set(self.prediction_history),
            key=self.prediction_history.count
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            probabilities = self.model.predict_proba(features)

        return stable_prediction, confidence
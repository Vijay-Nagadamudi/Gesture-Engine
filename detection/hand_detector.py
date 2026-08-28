import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Current folder:", os.getcwd())
print("Model exists:", os.path.exists("models/hand_landmarker.task"))
class HandDetector:
    def __init__(self):

        with open("models/hand_landmarker.task", "rb") as f:
            model_data = f.read()

        base_options = python.BaseOptions(
            model_asset_buffer = model_data
        )

        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = 1
        )

        self.hands = vision.HandLandmarker.create_from_options(options)
        
    
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(
            image_format = mp.ImageFormat.SRGB,
            data = rgb_frame
        )
        results = self.hands.detect(mp_image)
        return results

    def close(self):
        self.hands.close()
        
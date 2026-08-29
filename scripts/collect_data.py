import cv2
import mediapipe as mp 
import csv
import os
import time

from camera.camera import Camera
from detection.hand_detector import HandDetector
from visualization.renderer import HandRenderer
from features.extractor import LandMarkFeatureExtractor

GESTURES = {
    "1" : "PINCH",
    "2" : "ITALIAN",
    "3" : "ILY",
    "4" : "ROCK",
    "5" : "CALL_ME",
    "6" : "FINGER_HEART",
    "7" : "OPEN_PALM",
    "8" : "FIST",
    "9" : "ONE",
    "0" : "TWO",
}

SAMPLES_PER_GESTURE = 500
DATASET_PATH = "data/gesture_dataset.csv" 

def save_sample(features,label):
    file_exists = os.path.exists(DATASET_PATH)
    
    with open(DATASET_PATH, "a",newline="") as file:
        writer = csv.writer(file)
        
        if not file_exists:
            header = [f"feature_{i}" for i in range(len(features))]
            header.append("label")
            writer.writerow(header)
        
        writer.writerow([*features, label])
        
def main():
    camera = Camera()
    detector = HandDetector()
    extractor = LandMarkFeatureExtractor()
    renderer = HandRenderer()
    
    os.makedirs("data", exist_ok=True)

    print("\nSelect a gesture:")
    for key, gesture in GESTURES.items():
        print(f"{key} → {gesture}")

    selected_key = input("\nEnter gesture number: ")

    if selected_key not in GESTURES:
        print("Invalid gesture selection.")
        return
    
    gesture = GESTURES[selected_key]

    print(f"\nGet ready for: {gesture}")
    print("Starting collection in 3....2.....1....GO!!")
    time.sleep(3)
    
    samples_collected = 0
    frame_count = 0
    
    while samples_collected < SAMPLES_PER_GESTURE:

        frame = camera.read()
        results = detector.detect(frame)

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]
            frame = renderer.draw_landmarks(frame, landmarks)

            frame_count += 1
            
            if frame_count % 10 == 0:
                features = extractor.extract(landmarks)
                save_sample(features,gesture)
                samples_collected += 1

                print(
                    f"\r{gesture}: "
                    f"{samples_collected}/{SAMPLES_PER_GESTURE}",
                    end=""
                )

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    camera.release()
    cv2.destroyAllWindows()

    print(f"\n\nFinished collecting {gesture} samples.")
    


if __name__ == "__main__":
    main()
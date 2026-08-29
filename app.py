import cv2

from camera.camera import Camera
from detection.hand_detector import HandDetector
from features.extractor import LandMarkFeatureExtractor
from models.predictor import GesturePredictor
from visualization.renderer import HandRenderer


def main():
    camera = Camera()
    detector = HandDetector()
    extractor = LandMarkFeatureExtractor()
    predictor = GesturePredictor()
    renderer = HandRenderer()

    while True:
        frame = camera.read()

        results = detector.detect(frame)

        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]

            frame = renderer.draw_landmarks(
                frame,
                landmarks
            )

            features = extractor.extract(landmarks)

            gesture, confidence = predictor.predict(features)
            
            cv2.putText(
                frame,
                f"Gesture : {gesture}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2   
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence : .0%}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("Gesture Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
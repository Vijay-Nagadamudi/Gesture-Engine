import cv2
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections



from camera.camera import Camera
from detection.hand_detector import HandDetector

def main():
    camera = Camera()
    detector = HandDetector()
    
    while True:
        frame = camera.read()
        results = detector.detect(frame)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = frame.shape

                # Draw landmarks
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw connections
                for connection in HandLandmarksConnections.HAND_CONNECTIONS:
                    start = hand_landmarks[connection.start]
                    end = hand_landmarks[connection.end]

                    start_point = (
                        int(start.x * w),
                        int(start.y * h)
                    )

                    end_point = (
                        int(end.x * w),
                        int(end.y * h)
                    )

                    cv2.line(
                        frame,
                        start_point,
                        end_point,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("Hand Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    detector.close()
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    
import cv2

from camera.camera import Camera

def main():
    camera = Camera()
    
    while True:
        frame = camera.read()
        cv2.imshow("Hand Gesture Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()    
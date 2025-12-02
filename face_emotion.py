# face_emotion.py
from deepface import DeepFace
import cv2

def start_face_emotion():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Cannot access webcam")
        return

    print("Press 'q' to quit the webcam window.")

    while True:
        ret, frame = cam.read()

        if not ret:
            print("Failed to grab frame")
            break

        try:
            # Analyze face emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']


            # Display emotion text on screen
            cv2.putText(frame, f"Emotion: {emotion}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        except Exception as e:
            print("Error:", e)

        cv2.imshow("Face Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_face_emotion()

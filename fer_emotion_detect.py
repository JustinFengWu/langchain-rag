# from fer import FER
# import cv2

# # Load an image using OpenCV
# img = cv2.imread("./pictures/angry_guy1.jpg")

# # Initialize the emotion detector
# detector = FER()

# # Detect the top emotion
# emotion, score = detector.top_emotion(img)

# # Output the result
# print(f"Detected Emotion: {emotion}, Score: {score}")


import cv2
from fer import FER

# Initialize the emotion detector
detector = FER()

# Capture webcam video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect the top emotion in the frame
    emotion, score = detector.top_emotion(frame)

    # Prepare the text to display
    if emotion is not None:
        emotion_text = f"{emotion}"
    else:
        emotion_text = "No emotion detected"

    # Show the detected emotion on the frame
    cv2.putText(frame, emotion_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

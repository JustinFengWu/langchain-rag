from fer import FER
import cv2

# Load an image using OpenCV
img = cv2.imread("./pictures/angry_guy1.jpg")

# Initialize the emotion detector
detector = FER()

# Detect the top emotion
emotion, score = detector.top_emotion(img)

# Output the result
print(f"Detected Emotion: {emotion}, Score: {score}")

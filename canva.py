import cv2
import  numpy as np
from HandTrackingModule import HandDetector
from transform import Transformation




cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
# Load background image
image_path = "image\image.png"  # Make sure this file is present in your working directory
background_img = cv2.imread(image_path)
BrushButton = [(0, 0), (200, 200)]
SquareButton = [(200, 0), (400, 200)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better interaction
    h, w, c = frame.shape

    # Resize background image to match the camera frame size
    if background_img is None:
        print("Error: Background image not loaded.")
        exit()
    background_resized = cv2.resize(background_img, (w, h))

    # Replace the entire camera feed with the image
    # frame = background_resized.copy()

    # Convert frame to RGB for MediaPipe processing
    
    _, frame = detector.findHands(img=frame, img_to_draw=background_resized, draw=True)
    # Show the updated frame

    cv2.imshow("Virtual Canvas", frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
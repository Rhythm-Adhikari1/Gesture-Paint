# import cv2
# import  numpy as np
# from HandTrackingModule import HandDetector
# from transform import Transformation

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# pen_mode = False
# writing_mode = False
# prev_x, prev_y = None, None

# canvas = None

# detector = HandDetector(detectionCon=0.8)
# # Load background image
# image_path = "image\image.png"  # Make sure this file is present in your working directory
# background_img = cv2.imread(image_path)
# BrushButton = [(0, 0), (200, 200)]
# SquareButton = [(200, 0), (400, 200)]

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)  # Flip for better interaction
#     h, w, c = frame.shape
    
    

#     # Resize background image to match the camera frame size
#     if background_img is None:
#         print("Error: Background image not loaded.")
#         exit()
#     background_resized = cv2.resize(background_img, (w, h))


#     if canvas is None:
#         canvas = np.zeros_like(background_resized)
        
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # result = hands.process(rgb_frame)
    
#     allHands, frame = detector.findHands(img=frame, img_to_draw=background_resized, draw=True)
#     # Show the updated frame
#     for hand in allHands:
#         if hand['type'] == 'Right':
#             lmlist = hand['lmlist']

#     cv2.imshow("Virtual Canvas", frame)

#     # Exit on pressing 'q'
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
from HandTrackingModule import HandDetector

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load Background Image
image_path = "image/image.png"  # Ensure the path is correct
background_img = cv2.imread(image_path)

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8)

# Persistent Drawing Canvas
canvas = None
pen_mode = False
writing_mode = False
prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for better interaction
    h, w, _ = frame.shape

    # Resize Background Image
    if background_img is None:
        print("Error: Background image not loaded.")
        exit()
    background_resized = cv2.resize(background_img, (w, h))

    # Initialize canvas if not already created
    if canvas is None:
        canvas = np.zeros_like(background_resized)  # Persistent drawing board

    # Detect Hands
    hands, frame = detector.findHands(frame, img_to_draw=background_resized)

    if hands:
        for hand in hands:
            # Get index and thumb tip positions
            lm_list = hand["lmList"]
            index_x, index_y = lm_list[8][:2]  # Index finger tip
            thumb_x, thumb_y = lm_list[4][:2]  # Thumb tip

            # Check for Pen Mode Activation (Pinch in Top-Left Corner)
            if index_x < 50 and index_y < 50 and thumb_x < 50 and thumb_y < 50:
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                if distance < 10:  # If fingers are close
                    pen_mode = True  # Enable Pen Mode
                    print("🖊 Pen Mode Activated")

            # Check for Writing Mode Activation (Pinch in White Area)
            if pen_mode:
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                if distance < 20:  # If fingers are pinched
                    writing_mode = True
                else:
                    writing_mode = False  # Turn off writing mode when fingers are released

            # Draw if Writing Mode is Active
            if writing_mode:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 255, 0), 5)
                prev_x, prev_y = index_x, index_y  # Update previous position
            else:
                prev_x, prev_y = None, None  # Reset previous position when not writing

    # Merge canvas with background to keep drawings persistent
    combined = cv2.addWeighted(background_resized, 0.8, canvas, 1, 0)

    # Show Output
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
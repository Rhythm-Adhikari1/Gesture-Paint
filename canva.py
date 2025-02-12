import cv2
import numpy as np
from HandTrackingModule import HandDetector
import time

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

# Square Selection
square_selected = False  # Flag for selecting a square
selected_square_index = None  # Stores index of the selected square
square_size = 50  # Default size of square
dropped_squares = []  # Stores dropped square positions
start_time = None  # Timer for square dropping

def handle_pinch(index_x, index_y, thumb_x, thumb_y):
    """
    Handles pinch gesture for selecting, moving, and dropping squares.
    """
    global square_selected, selected_square_index, start_time

    # Compute pinch distance
    distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))

    # Check if selecting a new square from the top-right corner
    if not square_selected and index_x > 1180 and index_y < 100 and thumb_x > 1180 and thumb_y < 100:
        if distance < 50:
            square_selected = True
            selected_square_index = None  # New square selection
            start_time = time.time()
            print("🟦 New Square Selected - Ready to Drop")

    # Check if selecting an already dropped square
    if not square_selected:
        for i, (sx, sy) in enumerate(dropped_squares):
            if sx <= index_x <= sx + square_size and sy <= index_y <= sy + square_size:
                if distance < 50:
                    square_selected = True
                    selected_square_index = i
                    start_time = time.time()
                    print("🔄 Square Re-selected - Move to New Place")
                    return

    # If a square is selected and 2 seconds have passed, drop it
    if square_selected and (time.time() - start_time) > 2 and distance < 50:
        new_x, new_y = index_x - square_size // 2, index_y - square_size // 2
        if selected_square_index is None:
            dropped_squares.append((new_x, new_y))  # Drop a new square
        else:
            dropped_squares[selected_square_index] = (new_x, new_y)  # Move existing square
        square_selected = False  # Reset selection
        print(f"📍 Square Dropped at ({new_x}, {new_y})")

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

            # Handle square selection, moving, and dropping
            handle_pinch(index_x, index_y, thumb_x, thumb_y)

    # Merge canvas with background to keep drawings persistent
    combined = cv2.addWeighted(background_resized, 0.8, canvas, 1, 0)

    # Draw all dropped squares
    for sx, sy in dropped_squares:
        cv2.rectangle(combined, (sx, sy), (sx + square_size, sy + square_size), (0, 255, 255), -1)

    # Show Output
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

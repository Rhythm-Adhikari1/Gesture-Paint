import cv2
import numpy as np
from HandTrackingModule import HandDetector

# Initialize Camera
cap = cv2.VideoCapture(0)
w, h = 1280, 720
cap.set(3, w)
cap.set(4, h)

# Load Background Image
image_path = "image/image.png"  # Ensure the path is correct
background_img = cv2.imread(image_path)

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8)

# Persistent Drawing Canvas
canvas = None
square_selected = False  # Flag for selecting a square
selected_square_index = None  # Stores index of the selected square
square_size = 50  # Default size of square
dropped_squares = []  # Stores dropped square positions
square_button_clicked = False  # Whether the square button is active
dragging = False  # Whether a square is being dragged
brush_button_clicked = False
prev_x, prev_y = None, None  # Previous coordinates for brush drawing

buttons = {
    "brush": [(0, 0), (200, 200)],
    "square": [(w - 200, 0), (w, 200)]
}

def is_index_up(hand):
    """ Returns True if only the index finger is up. """
    fingers_up = detector.fingersUp(hand)
    return fingers_up[0] == 0 and fingers_up[1] == 1 and all(f == 0 for f in fingers_up[2:])

def is_thumb_and_index_up(hand):
    """ Returns True if both the thumb and index finger are up. """
    fingers_up = detector.fingersUp(hand)
    return   all(f == 1 for f in fingers_up[:])

def square(index_x, index_y, hand):
    global square_selected, selected_square_index, square_button_clicked, dragging, brush_button_clicked

    # Square Button Coordinates
    x1s, y1s, x2s, y2s = *buttons["square"][0], *buttons["square"][1]

    # **Click Square Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            square_button_clicked = True
            brush_button_clicked = False
            print("🟦 Square Button Clicked")

    # **Drop a New Square (When Square Button is Active)**
    if square_button_clicked and is_index_up(hand) and not (x1s < index_x < x2s and y1s < index_y < y2s):
        new_x, new_y = index_x - square_size // 2, index_y - square_size // 2
        dropped_squares.append((new_x, new_y))  # Drop square
        square_button_clicked = False  # Reset button
        print(f"📍 Square Dropped at ({new_x}, {new_y})")
        return

    # **Select & Drag an Existing Square**
    if not dragging:
        for i, (sx, sy) in enumerate(dropped_squares):
            if sx <= index_x <= sx + square_size and sy <= index_y <= sy + square_size:
                if is_index_up(hand):
                    square_selected = True
                    selected_square_index = i
                    dragging = True  # Start dragging
                    print("🔄 Square Selected for Dragging")

    # **Move the Selected Square**
    if dragging and square_selected:
        new_x, new_y = index_x - square_size // 2, index_y - square_size // 2
        dropped_squares[selected_square_index] = (new_x, new_y)

        # **Release the Square When Both Thumb and Index Are Up**
        if is_thumb_and_index_up(hand):
            square_selected = False
            dragging = False
            print(f"✅ Square Dropped at ({new_x}, {new_y})")

def brush(index_x, index_y, hand):
    global brush_button_clicked, prev_x, prev_y, canvas

    x1b, y1b, x2b, y2b = *buttons["brush"][0], *buttons["brush"][1]

    # **Click Brush Button**
    if x1b < index_x < x2b and y1b < index_y < y2b:
        if is_index_up(hand):  # Only Index Finger Up
            brush_button_clicked = True
            print("🖌 Brush Button Clicked")

    # **Draw if Brush is Selected & Only Index Finger is Up**
    elif brush_button_clicked and is_index_up(hand):
        if prev_x is None or prev_y is None:
            prev_x, prev_y = index_x, index_y  # Initialize previous position

        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (0, 0, 255), 5)  # Draw red line
        prev_x, prev_y = index_x, index_y  # Update previous position

    else:
        prev_x, prev_y = None, None  # Reset drawing state when fingers are not in the right position


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
            lm_list = hand["lmList"]
            index_x, index_y = lm_list[8][:2]  # Index finger tip
            square(index_x, index_y, hand)
            brush(index_x, index_y, hand)

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

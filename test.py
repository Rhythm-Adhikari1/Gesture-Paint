import cv2
import numpy as np
from HandTrackingModule import HandDetector
from shapes import Draw

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
draw_shapes = Draw()

# Persistent Drawing Canvas
canvas = np.zeros((h, w, 3), dtype=np.uint8)

dropped_shapes = []  # Stores dropped square positions
initial_square_coordinates = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
prev_index_x, prev_index_y = None, None  # Previous index finger coordinates
square_button_clicked = False
brush_button_clicked = False

# Define Buttons
buttons = {
    "brush": [(0, 0), (200, 200)],
    "square": [(w - 200, 0), (w, 200)],
}

def is_index_up(hand):
    """ Returns True if only the index finger is up. """
    fingers_up = detector.fingersUp(hand)
    return fingers_up[1] == 1 and all(f == 0 for f in fingers_up if f != fingers_up[1])

def square(index_x, index_y, hand):
    global square_button_clicked
    x1, y1, x2, y2 = *buttons["square"][0], *buttons["square"][1]

    # Click Square Button
    if x1 < index_x < x2 and y1 < index_y < y2:
        if is_index_up(hand):
            square_button_clicked = True
            print("🟦 Square Button Clicked")

    # Drop Square
    elif square_button_clicked and is_index_up(hand):
        updated_coordinates = [(x + index_x, y + index_y) for x, y in initial_square_coordinates]
        dropped_shapes.append(updated_coordinates)
        square_button_clicked = False  # Reset button
        print(f"📍 Square Dropped at ({index_x}, {index_y})")

def brush(index_x, index_y, hand):
    global brush_button_clicked, prev_index_x, prev_index_y, canvas
    x1, y1, x2, y2 = *buttons["brush"][0], *buttons["brush"][1]

    # Click Brush Button
    if x1 < index_x < x2 and y1 < index_y < y2:
        if is_index_up(hand):
            brush_button_clicked = True
            print("🖌 Brush Button Clicked")

    # Draw if Brush is Selected
    elif brush_button_clicked and is_index_up(hand):
        if prev_index_x is None or prev_index_y is None:
            prev_index_x, prev_index_y = index_x, index_y
        cv2.line(canvas, (prev_index_x, prev_index_y), (index_x, index_y), (0, 0, 255), 5)
        prev_index_x, prev_index_y = index_x, index_y
    else:
        prev_index_x, prev_index_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    background_resized = cv2.resize(background_img, (w, h)) if background_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
    hands, frame = detector.findHands(frame, img_to_draw=background_resized)

    if hands:
        for hand in hands:
            lm_list = hand["lmList"]
            index_x, index_y = lm_list[8][:2]
            square(index_x, index_y, hand)
            brush(index_x, index_y, hand)

    combined = cv2.addWeighted(background_resized, 0.8, canvas, 1, 0)
    for shape_coordinates in dropped_shapes:
        draw_shapes.polygon(combined, points=shape_coordinates, color=(0, 255, 0))
    
    cv2.imshow("Virtual Canvas", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

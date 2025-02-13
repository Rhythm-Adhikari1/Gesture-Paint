import cv2
import numpy as np
from HandTrackingModule import HandDetector
from shapes import Draw
from clipping import clip_polygon

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
canvas = None
square_selected = False           # (Not used in new dragging logic)
selected_square_index = None      # Index of the currently selected polygon for dragging
square_size = 50                  # Default size of square
dropped_shapes = []               # List to store dropped square (polygon) coordinates
square_button_clicked = False     # Whether the square button is active
dragging = False                  # (Not used in new dragging logic)
brush_button_clicked = False
prev_x, prev_y = None, None       # Previous coordinates for brush drawing
initial_square_coordinates = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
prev_index_x, prev_index_y = None, None  # Previous index finger coordinates for dragging
clip_rect = [(50, 200), (w - 250, 200), (w - 250, h - 50), (50, h - 50)]

# Define button areas
buttons = {
    "brush": [(0, 0), (200, 200)],
    "square": [(w - 200, 0), (w, 200)],
    "rectangle": [(w - 400, 0), (w - 200, 200)],
    "circle": [(w - 600, 0), (w - 400, 200)],
    "triangle": [(w - 800, 0), (w - 600, 200)],
}

def is_index_up(hand):
    """Returns True if only the index finger is up."""
    fingers_up = detector.fingersUp(hand)
    return fingers_up[0] == 0 and fingers_up[1] == 1 and all(f == 0 for f in fingers_up[2:])

def is_all_finger_up(hand):
    """Returns True if all fingers are up."""
    fingers_up = detector.fingersUp(hand)
    return all(f == 1 for f in fingers_up)

def polygon(index_x, index_y, hand):
    # Placeholder for polygon creation (if needed)
    pass

def square(index_x, index_y, hand):
    global square_button_clicked, brush_button_clicked

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
        updated_coordinates = [(x + index_x, y + index_y) for x, y in initial_square_coordinates]
        dropped_shapes.append(updated_coordinates)  # Add new square to the list
        square_button_clicked = False  # Reset button
        print("📍 Square Dropped")

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
        prev_x, prev_y = None, None  # Reset when not drawing

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

            # Process button actions (create square, brush, etc.)
            square(index_x, index_y, hand)
            brush(index_x, index_y, hand)

            # -----------------------------
            # DRAGGING LOGIC (Selection & Movement)
            # -----------------------------
            # When the index finger is up, we either select a polygon or move the selected one.
            if is_index_up(hand):
                # If a polygon is already selected, move it
                if selected_square_index is not None:
                    if prev_index_x is None or prev_index_y is None:
                        prev_index_x, prev_index_y = index_x, index_y
                    dx = index_x - prev_index_x
                    dy = index_y - prev_index_y
                    # Update the selected polygon's coordinates
                    dropped_shapes[selected_square_index] = [
                        (x + dx, y + dy) for (x, y) in dropped_shapes[selected_square_index]
                    ]
                    prev_index_x, prev_index_y = index_x, index_y
                else:
                    # No polygon selected yet: try to select one
                    # Check in reverse order (assume last drawn is on top)
                    for i in range(len(dropped_shapes) - 1, -1, -1):
                        coordinates_numpy = np.array(dropped_shapes[i], np.int32)
                        if cv2.pointPolygonTest(coordinates_numpy, (index_x, index_y), False) >= 0:
                            selected_square_index = i
                            prev_index_x, prev_index_y = index_x, index_y
                            print(f"🔄 Polygon {i} selected for dragging")
                            break
            else:
                # If index finger is not up, clear the selection
                selected_square_index = None
                prev_index_x, prev_index_y = None, None

    # Merge canvas with background to keep drawings persistent
    combined = cv2.addWeighted(background_resized, 0.8, canvas, 1, 0)

    # Draw all polygons (after clipping)
    for shape in dropped_shapes:
        if shape:
            clipped_shape = clip_polygon(shape, clip_rect)
            if clipped_shape:
                draw_shapes.polygon(combined, points=clipped_shape, color=(0, 255, 0))

    # Show Output
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

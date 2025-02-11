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

# Square Selection and Dragging
square_selected = False
square_x, square_y = 300, 200  # Default square position
square_size = 50
dragging_square = False

# Initialize a list to store the positions of all dropped squares
# Initialize a list to store the positions of all dropped squares
dropped_squares = []

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

        #    Check for Square Selection Mode (Pinch near Top-Right Corner)
            if index_x > w - 100 and index_y < 100 and thumb_x > w - 100 and thumb_y < 100:
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                if distance < 50:  # If fingers are close
                    square_selected = True  # Enable Square Selection
                    dragging_square = False  # Start Dragging
                    print("🟦 Square Mode Activated")
                elif distance > 50:
                    dragging_square = True  # Dragging Mode Active
            
            # Dragging Mode
            if dragging_square:
                # Update the square's coordinates
                square_x, square_y = index_x - square_size // 2, index_y - square_size // 2

                # Check if the pinch is released to settle the square
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array([thumb_x, thumb_y]))
                if distance < 40:  # Pinch released
                    dragging_square = False  # Stop dragging
                    print("✅ Square Settled")

                    # If the square was dropped at a new location, add it to the list
                    dropped_squares.append((square_x, square_y))
                    print(f"📍 Square Dropped at ({square_x}, {square_y})")

    # Merge canvas with background to keep drawings persistent
    combined = cv2.addWeighted(background_resized, 0.8, canvas, 1, 0)

    # Draw all the dropped squares
    for square in dropped_squares:
        sx, sy = square
        cv2.rectangle(combined, (sx, sy), (sx + square_size, sy + square_size), (0, 255, 255), -1)

    # If dragging, draw the square at the current position
    if dragging_square:
        cv2.rectangle(combined, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 255), -1)

    # Show Output
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

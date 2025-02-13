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
rectangle_button_clicked = False  # Whether the rectangle button is active
triangle_button_clicked = False   # Whether the triangle button is active
line_button_clicked = False       # Whether the line button is active
circle_button_clicked = False     # Whether the circle button is active
dragging = False                  # (Not used in new dragging logic)
brush_button_clicked = False
prev_x, prev_y = None, None       # Previous coordinates for brush drawing
initial_square_coordinates = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
prev_index_x, prev_index_y = None, None  # Previous index finger coordinates for dragging
drawing_canvas = [(30, 150), (1030, 150), (1030, h - 50), (30, h - 50)]
red_color_clicked = False
blue_color_clicked = False
green_color_clicked = False
yellow_color_clicked = False



# Define button areas
buttons = {
    "brush": [(20, 20), (100, 100)],
    "eraser": [(120, 20), (200, 100)],
    "square": [(620,20), (685, 95)],
    "rectangle": [(700, 20), (775 , 95)],
    "line": [(785, 20), (855, 95)],
    "triangle": [(870, 20), (945 , 95)],
    "circle" : [(955, 20), (1030, 95)],
    "red": [(230, 20), (310, 100)],
    "blue": [(325, 20), (495, 100)],
    "green": [(380, 20), (485, 100)],
    "yellow": [(495, 20), (575, 100)],
    "undo" : [(865, 20), (950, 100)],
    "redo" : [(1070, 20), (1150, 100)],
}

RED_COLOR_CODE = (0, 0, 255)       # Red in BGR
BLUE_COLOR_CODE = (255, 0, 0)      # Blue in BGR
GREEN_COLOR_CODE = (0, 255, 0)     # Green in BGR
YELLOW_COLOR_CODE = (0, 255, 255)  # Yellow in BGR


# disable other colors when selecting a color
def select_color(color):
    global red_color_clicked, blue_color_clicked, green_color_clicked, yellow_color_clicked
    if(color == "red"):
        blue_color_clicked = False
        green_color_clicked = False
        yellow_color_clicked = False
    elif(color == "blue"):
        red_color_clicked = False
        green_color_clicked = False
        yellow_color_clicked = False
    elif(color == "yellow"):
        red_color_clicked = False
        green_color_clicked = False
        blue_color_clicked = False
    elif(color == "green"):
        red_color_clicked = False
        blue_color_clicked = False
        yellow_color_clicked = False

# determine which color is selected at present
def get_color():
    global red_color_clicked, blue_color_clicked, green_color_clicked, yellow_color_clicked
    if(red_color_clicked):
        return RED_COLOR_CODE
    elif(blue_color_clicked):
        return BLUE_COLOR_CODE
    elif(green_color_clicked):
        return GREEN_COLOR_CODE
    elif(yellow_color_clicked):
        return YELLOW_COLOR_CODE
    else:
        return RED_COLOR_CODE


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
        
def rectangle(index_x, index_y, hand):
    global rectangle_button_clicked, brush_button_clicked

    # rectangle Button Coordinates
    x1s, y1s, x2s, y2s = *buttons["rectangle"][0], *buttons["rectangle"][1]

    # **Click rectangle Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            rectangle_button_clicked = True
            brush_button_clicked = False
            print("🟦 rectangle Button Clicked")

    # **Drop a New rectangle (When rectangle Button is Active)**
    if rectangle_button_clicked and is_index_up(hand) and not (x1s < index_x < x2s and y1s < index_y < y2s):
        updated_coordinates = [(x + index_x, y + index_y) for x, y in initial_square_coordinates]
        dropped_shapes.append(updated_coordinates)  # Add new rectangle to the list
        rectangle_button_clicked = False  # Reset button
        print("📍 rectangle Dropped")


def triangle(index_x, index_y, hand):
    global triangle_button_clicked, brush_button_clicked

    # Triangle Button Coordinates
    x1t, y1t, x2t, y2t = *buttons["triangle"][0], *buttons["triangle"][1]

    # **Click Triangle Button (Only Index Finger Up)**
    if x1t < index_x < x2t and y1t < index_y < y2t:
        if is_index_up(hand):
            triangle_button_clicked = True
            brush_button_clicked = False
            print("🔺 Triangle Button Clicked")

    # **Drop a New Triangle (When Triangle Button is Active)**
    if triangle_button_clicked and is_index_up(hand) and not (x1t < index_x < x2t and y1t < index_y < y2t):
        triangle_size = 50  # Adjust size as needed
        updated_triangle_coordinates = [
            (index_x, index_y - triangle_size),  # Top vertex
            (index_x - triangle_size, index_y + triangle_size),  # Bottom left vertex
            (index_x + triangle_size, index_y + triangle_size)   # Bottom right vertex
        ]
        dropped_shapes.append(updated_triangle_coordinates)  # Add new triangle to the list
        triangle_button_clicked = False  # Reset button
        print("📍 Triangle Dropped")
        



def draw_line(index_x, index_y, hand):
    global line_button_clicked, brush_button_clicked

    # Line Button Coordinates
    x1l, y1l, x2l, y2l = *buttons["line"][0], *buttons["line"][1]

    # **Click Line Button (Only Index Finger Up)**
    if x1l < index_x < x2l and y1l < index_y < y2l:
        if is_index_up(hand):
            line_button_clicked = True
            brush_button_clicked = False
            print("📏 Line Button Clicked")

    # **Drop a New Line (When Line Button is Active)**
    if line_button_clicked and is_index_up(hand) and not (x1l < index_x < x2l and y1l < index_y < y2l):
        line_length = 80  # Adjust as needed
        updated_line_coordinates = [
            (index_x - line_length // 2, index_y),  # Start point (Left)
            (index_x + line_length // 2, index_y)   # End point (Right)
        ]
        dropped_shapes.append(updated_line_coordinates)  # Add new line to the list
        line_button_clicked = False  # Reset button
        print("📍 Line Dropped")
        
        
def draw_circle(index_x, index_y, hand):
    global circle_button_clicked, brush_button_clicked

    # Circle Button Coordinates
    x1c, y1c, x2c, y2c = *buttons["circle"][0], *buttons["circle"][1]

    # **Click Circle Button (Only Index Finger Up)**
    if x1c < index_x < x2c and y1c < index_y < y2c:
        if is_index_up(hand):
            circle_button_clicked = True
            brush_button_clicked = False
            print("⭕ Circle Button Clicked")

    # **Drop a New Circle (When Circle Button is Active)**
    if circle_button_clicked and is_index_up(hand) and not (x1c < index_x < x2c and y1c < index_y < y2c):
        circle_radius = 40  # Adjust radius as needed
        updated_circle_coordinates = (index_x, index_y, circle_radius)  # (CenterX, CenterY, Radius)
        dropped_shapes.append(updated_circle_coordinates)  # Add new circle to the list
        circle_button_clicked = False  # Reset button
        print("📍 Circle Dropped")
        

        
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
        cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), get_color(), 5)  # Draw red line
        prev_x, prev_y = index_x, index_y  # Update previous position
    else:
        prev_x, prev_y = None, None  # Reset when not drawing

def red(index_x, index_y, hand):
    global red_color_clicked, brush_button_clicked

    #red button coordinates
    x1s, y1s, x2s, y2s = *buttons["red"][0], *buttons["red"][1]

    # **Click Square Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            red_color_clicked = True
            select_color("red")
            print("🟦 Red color Clicked")

def blue(index_x, index_y, hand):
    global blue_color_clicked, brush_button_clicked

    #red button coordinates
    x1s, y1s, x2s, y2s = *buttons["blue"][0], *buttons["blue"][1]

    # **Click Square Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            blue_color_clicked = True
            select_color("blue")
            print("🟦 Blue color Clicked")

def green(index_x, index_y, hand):
    global green_color_clicked, brush_button_clicked

    #red button coordinates
    x1s, y1s, x2s, y2s = *buttons["green"][0], *buttons["green"][1]

    # **Click Square Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            green_color_clicked = True
            select_color("green")
            print("🟦 Green color Clicked")

def yellow(index_x, index_y, hand):
    global yellow_color_clicked, brush_button_clicked

    #red button coordinates
    x1s, y1s, x2s, y2s = *buttons["yellow"][0], *buttons["yellow"][1]

    # **Click Square Button (Only Index Finger Up)**
    if x1s < index_x < x2s and y1s < index_y < y2s:
        if is_index_up(hand):
            yellow_color_clicked = True
            select_color("yellow")
            print("🟦 Yellow color Clicked")



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

            red(index_x, index_y, hand)
            blue(index_x, index_y, hand)
            green(index_x, index_y, hand)
            yellow(index_x, index_y, hand)
            
            rectangle(index_x, index_y, hand)
            triangle(index_x, index_y, hand)
            draw_line(index_x, index_y, hand)
            draw_circle(index_x, index_y, hand)

            #print cooridinates
            # print(index_x, index_y)

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
            clipped_shape = clip_polygon(shape, drawing_canvas)
            if clipped_shape:
                draw_shapes.polygon(combined, points=clipped_shape, color=(0, 255, 0))

    # Show Output
    cv2.imshow("Virtual Canvas", combined)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
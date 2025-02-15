import cv2
import numpy as np
from HandTrackingModule import HandDetector
from shapes import Draw
from clipping import clip_polygon
import math
from transform import Transformation
from fill import Fill
import copy
class DrawingApp:
    def __init__(self):
        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        self.w, self.h = 1280, 720
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)

        # Load Background Image
        self.background_img = cv2.imread("image/image.png")
        if self.background_img is None:
            print("Error: Background image not loaded.")
            exit(1)

        # Initialize Hand Detector and drawing tool
        self.detector = HandDetector(detectionCon=0.8)
        self.transformation = Transformation()
        self.draw_shapes = Draw()

        # Persistent drawing canvas and dropped shapes
        self.canvas = None
        self.dropped_shapes = []
        self.initial_square_coordinates = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
        self.default_color = (0, 0, 255)  # Default color is red

        # Tool button states stored in dictionaries
        self.shape_flags = {
            "square": False,
            "rectangle": False,
            "triangle": False,
            "line": False,
            "circle": False,
        }
        self.brush_button_clicked = False

        # Color buttons (using dictionary for simplicity)
        self.color_flags = {
            "red": False,
            "blue": False,
            "green": False,
            "yellow": False,
        }
        self.color_map = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
        }

        # Variables for brush drawing
        self.prev_x, self.prev_y = None, None

        # Variables for dragging/resizing shapes
        self.selected_shape_index = None
        self.prev_index_x, self.prev_index_y = None, None
        self.prev_angle = None
        self.rotation = False
        self.rot_selected_shape_index = None
        self.scaling_selected_shape_index = None
        self.initial_scaling_distance = None
        self.original_shape_for_scaling = None
        self.scaling = False
        self.fill_selected_shape_index = None
        self.filling = False
        self.original_shape_for_filling = None

        # Define button areas as (top-left, bottom-right)
        self.buttons = {
            "brush": [(20, 20), (100, 100)],
            "eraser": [(120, 20), (200, 100)],
            "square": [(620, 20), (685, 95)],
            "rectangle": [(700, 20), (775, 95)],
            "line": [(785, 20), (855, 95)],
            "triangle": [(870, 20), (945, 95)],
            "circle": [(955, 20), (1030, 95)],
            "red": [(230, 20), (310, 100)],
            "blue": [(325, 20), (495, 100)],
            "green": [(380, 20), (485, 100)],
            "yellow": [(495, 20), (575, 100)],
            "undo": [(865, 20), (950, 100)],
            "redo": [(1070, 20), (1150, 100)],
        }

    # ---------------------- Main Loop ----------------------
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.h, self.w, _ = frame.shape

            # Define drawing area and background
            drawing_canvas = [(30, 150), (1030, 150), (1030, self.h - 50), (30, self.h - 50)]
            background_resized = cv2.resize(self.background_img, (self.w, self.h))
            if self.canvas is None:
                self.canvas = np.zeros_like(background_resized)

            # Det print("here")ect hands
            hands, frame = self.detector.findHands(frame, img_to_draw=background_resized)
           
            

            if hands:
                drawing_hand = None
                helping_hand = None

                

                if len(hands) == 1:
                    drawing_hand = hands[0]
                elif len(hands) == 2:

                    for hand in hands:
                        if hand["type"] == "Left":
                            drawing_hand = hand
                        else:
                            helping_hand = hand
                
                if drawing_hand and helping_hand:
                    self.handle_reshaping(drawing_hand, helping_hand)
                    self.handle_rotating(drawing_hand, helping_hand)
                    self.handle_fill(drawing_hand, helping_hand)
                
                if drawing_hand:
                    self.process_hand_buttons(drawing_hand)
                    self.handle_dragging(drawing_hand)
                


            # Merge canvas and render shapes
            combined = self.render_canvas(background_resized, drawing_canvas)
            cv2.imshow("Virtual Canvas", combined)
            self.rotation = False

            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ---------------------- Helper Methods ----------------------
    def point_in_rect(self, x, y, rect):
        (x1, y1), (x2, y2) = rect
        return x1 < x < x2 and y1 < y < y2

    def check_hand_inside_canvas(self, x, y):
        return 30 < x < 1030 and 150 < y < (self.h - 50)

    def is_index_up(self, hand):
        """Return True if only the index finger is up."""
        fingers_up = self.detector.fingersUp(hand)
        return fingers_up[0] == 0 and fingers_up[1] == 1 and all(f == 0 for f in fingers_up[2:])

    def get_color(self):
        for color, flag in self.color_flags.items():
            if flag:
                return self.color_map[color]
        return self.color_map["red"]

    def render_canvas(self, background, drawing_canvas):
        combined = cv2.addWeighted(background, 0.8, self.canvas, 1, 0)
        for shape, color in self.dropped_shapes:
            if shape:
                if isinstance(shape, tuple):
                    # Handle circle drawing
                    cx, cy, r = shape
                    cv2.circle(combined, (int(cx), int(cy)), int(r), color, 1)
                else:
                    # Handle polygon shapes
                    clipped = clip_polygon(shape, drawing_canvas)
                    if clipped:
                        self.draw_shapes.polygon(combined, points=clipped, color=color)
        return combined

    def select_shape_at(self, x, y):
        """Return the index of a shape that contains point (x,y), or None."""
        for i in range(len(self.dropped_shapes) - 1, -1, -1):
            shape_info = self.dropped_shapes[i]
            shape = shape_info[0]
            if isinstance(shape, tuple):
                if self.is_point_in_circle(x, y, shape):
                    print(f"Circle {i} selected for dragging")
                    return i
            else:
                pts = np.array(shape, np.int32)
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    print(f"Shape {i} selected for dragging")
                    return i
        return None

    def is_point_in_circle(self, x, y, circle):
        cx, cy, r = circle
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    # ---------------------- Generic Tool Handlers ----------------------
    def handle_shape_button(self, tool, drop_func, x, y, hand):
        """Generic handler for shape buttons (square, rectangle, etc.)."""
        rect = self.buttons[tool]
        if self.point_in_rect(x, y, rect):
            if self.is_index_up(hand):
                self.shape_flags[tool] = True
                self.brush_button_clicked = False
                print(f"{tool.capitalize()} Button Clicked")
        elif self.shape_flags.get(tool, False) and self.is_index_up(hand):
            self.dropped_shapes.append(drop_func(x, y))
            self.shape_flags[tool] = False
            print(f"{tool.capitalize()} Dropped")

    def handle_color_button(self, color, x, y, hand):
        """Generic handler for color buttons."""
        rect = self.buttons[color]
        if self.point_in_rect(x, y, rect) and self.is_index_up(hand):
            for c in self.color_flags:
                self.color_flags[c] = (c == color)
            print(f"{color.capitalize()} Color Clicked")

    def handle_brush(self, x, y, hand):
        rect = self.buttons["brush"]
        if self.point_in_rect(x, y, rect):
            if self.is_index_up(hand):
                self.brush_button_clicked = True
                print("Brush Button Clicked")
        elif self.brush_button_clicked and self.is_index_up(hand) and self.check_hand_inside_canvas(x, y):
            if self.prev_x is None or self.prev_y is None:
                self.prev_x, self.prev_y = x, y
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.get_color(), 5)
            self.prev_x, self.prev_y = x, y
        else:
            self.prev_x, self.prev_y = None, None

    # ---------------------- Drop Functions for Shapes ----------------------
    def drop_square(self, x, y):
        return ([(dx + x, dy + y) for dx, dy in self.initial_square_coordinates], self.get_color())

    def drop_rectangle(self, x, y):
        return ([(dx + x, dy + y) for dx, dy in self.initial_square_coordinates], self.get_color())

    def drop_triangle(self, x, y):
        triangle_size = 50
        return ([(x, y - triangle_size),
                (x - triangle_size, y + triangle_size),
                (x + triangle_size, y + triangle_size)], self.get_color())

    def drop_line(self, x, y):
        line_length = 80
        return ([(x - line_length // 2, y), (x + line_length // 2, y)], self.get_color())

    def drop_circle(self, x, y):
        circle_radius = 40
        return (x, y, circle_radius), self.get_color()

    # ---------------------- Process Hand Buttons ----------------------
    def process_hand_buttons(self, hand):
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]
        # Process shape buttons
        self.handle_shape_button("square", self.drop_square, x, y, hand)
        self.handle_shape_button("rectangle", self.drop_rectangle, x, y, hand)
        self.handle_shape_button("triangle", self.drop_triangle, x, y, hand)
        self.handle_shape_button("line", self.drop_line, x, y, hand)
        self.handle_shape_button("circle", self.drop_circle, x, y, hand)
        # Process brush tool
        self.handle_brush(x, y, hand)
        # Process color buttons
        self.handle_color_button("red", x, y, hand)
        self.handle_color_button("blue", x, y, hand)
        self.handle_color_button("green", x, y, hand)
        self.handle_color_button("yellow", x, y, hand)

    # ---------------------- Dragging & Reshaping ----------------------
    def handle_dragging(self, hand):


        if self.rotation:
            return
        
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]



        if self.is_index_up(hand):
            if self.selected_shape_index is not None:
                #if self.rotation == True:
                #    return
                
                # Drag the selected shape

                
                
                if self.prev_index_x is None or self.prev_index_y is None:
                    self.prev_index_x, self.prev_index_y = x, y
                dx = x - self.prev_index_x
                dy = y - self.prev_index_y
                shape, color = self.dropped_shapes[self.selected_shape_index]
                if isinstance(shape, tuple):
                    cx, cy, r = shape
                    self.dropped_shapes[self.selected_shape_index] = ((cx + dx, cy + dy, r), color)
                else:
                    self.dropped_shapes[self.selected_shape_index] = ([(px + dx, py + dy) for (px, py) in shape], color)
                self.prev_index_x, self.prev_index_y = x, y
            else:
                # Try to select a shape under the finger
                self.selected_shape_index = self.select_shape_at(x, y)
                if self.selected_shape_index is not None:
                    self.prev_index_x, self.prev_index_y = x, y
        else:
            self.selected_shape_index = None
            self.prev_index_x, self.prev_index_y = None, None

    
    def is_index_and_thump_up(self,hand):
        fingers = self.detector.fingersUp(hand)
        return fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] ==0


    

    def handle_reshaping(self, right_hand, left_hand):
    # Check for the correct scaling gesture
        if self.is_index_up(right_hand) and self.is_index_and_thump_up(left_hand):
            l_lmList = left_hand["lmList"]
            r_lmList = right_hand["lmList"]

            # Get the right-hand index tip coordinates
            px, py = r_lmList[8][:2]

            # If no shape is currently selected for scaling, select one using the right-hand tip
            if self.scaling_selected_shape_index is None:
                self.scaling_selected_shape_index = self.select_shape_at(px, py)
                self.initial_scaling_distance = None
                self.original_shape_for_scaling = None

            if self.scaling_selected_shape_index is not None:
                # Compute the current distance between left-hand index and thumb tips
                index_tip = np.array(l_lmList[8][:2])
                thumb_tip = np.array(l_lmList[4][:2])
                current_distance = np.linalg.norm(index_tip - thumb_tip)

                # On the first frame of the gesture, store initial values
                if self.initial_scaling_distance is None:
                    self.initial_scaling_distance = current_distance
                    self.original_shape_for_scaling = copy.deepcopy(
                        self.dropped_shapes[self.scaling_selected_shape_index]
                    )

                # Calculate scaling ratio
                scaling_ratio = current_distance / self.initial_scaling_distance

                selected_shape, color = self.original_shape_for_scaling

                # Handle circles differently from polygons
                if isinstance(selected_shape, tuple):
                    # For circles: (cx, cy, r)
                    cx, cy, original_radius = selected_shape
                    new_radius = int(original_radius * scaling_ratio)
                    self.dropped_shapes[self.scaling_selected_shape_index] = ((cx, cy, new_radius), color)
                else:
                    # For polygons
                    pts = np.array(selected_shape)
                    pivx, pivy = np.mean(pts, axis=0)
                    scaled_shape = self.transformation.scale(
                        points=selected_shape,
                        pivot=(pivx, pivy),
                        sx=scaling_ratio,
                        sy=scaling_ratio
                    )
                    self.dropped_shapes[self.scaling_selected_shape_index] = (scaled_shape, color)
        else:
            # Reset scaling state when gesture ends
            self.scaling_selected_shape_index = None
            self.initial_scaling_distance = None
            self.original_shape_for_scaling = None
            self.scaling = False

    def handle_fill(self, right_hand, left_hand):

    # Check for the correct filling gesture
        if self.is_index_up(right_hand) and self.index_and_middle_up(left_hand):
            l_lmList = left_hand["lmList"]
            r_lmList = right_hand["lmList"] 

            # Get the right-hand index tip coordinates
            px, py = r_lmList[8][:2]

            # If no shape is currently selected for scaling, select one using the right-hand tip
            if self.fill_selected_shape_index is None:
                self.fill_selected_shape_index = self.select_shape_at(px, py)
                

            if self.fill_selected_shape_index is not None:
                self.original_shape_for_filling = copy.deepcopy(
                        self.dropped_shapes[self.fill_selected_shape_index]
                    )
                Fill.scanline_fill(self, self.original_shape_for_filling, self.get_color())

               
        else:
            # Reset fill state when gesture ends
            self.fill_selected_shape_index = None
            self.filling = False
            self.original_shape_for_filling = None


    def is_hand_closed(self, hand):
        fingers_up = self.detector.fingersUp(hand)
        return sum(fingers_up) == 0
    
    def index_and_middle_up(self, hand):
        fingers_up = self.detector.fingersUp(hand)
        return fingers_up[0] == 0 and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 0 and fingers_up[4] ==0
    
    
    def handle_rotating(self, right_hand, left_hand):
                
        if self.is_index_up(right_hand) and self.is_hand_closed(left_hand):

            l_lmList = left_hand["lmList"]
            r_lmList = right_hand["lmList"]
            print("Rotating gesture detected")

            px, py = r_lmList[8][:2]
            wx, wy, _ = l_lmList[0]  
            mx, my, _ = l_lmList[9]  

            # Compute the current angle (in radians)
            angle = math.atan2(-my + wy, mx - wx)
          
            if self.rot_selected_shape_index is not None:

                self.rotation = True

                print(f"Shape {self.rot_selected_shape_index} selected for rotating")

                if self.prev_angle is None:
                    self.prev_angle = angle  # Set initial angle

                # Calculate angle difference (in radians)
                dtheta = angle - self.prev_angle  
                print(dtheta)
                # Convert angle difference to degrees
                dtheta_deg = np.degrees(dtheta)
                sensitivity = 2  # Adjust sensitivity multiplier as needed

                selected_shape, color = self.dropped_shapes[self.rot_selected_shape_index]
                if isinstance(selected_shape, tuple):
                    return  # Skip circles for rotation

                print(selected_shape)
                for i in range(len(selected_shape)):
                    selected_shape[i] = (selected_shape[i][0], -selected_shape[i][1])

                # Compute the centroid of the shape (pivot remains constant)
                pivx, pivy = 0, 0
                for x, y in selected_shape:
                    pivx += x
                    pivy += y
                pivx /= len(selected_shape)
                pivy /= len(selected_shape)

                # Rotate the shape by the computed degree difference times sensitivity
                rotated_shape = self.transformation.rotate(selected_shape, (pivx, pivy), dtheta_deg * sensitivity)

                # Update the shape with its rotated coordinates

                for i in range(len(rotated_shape)):
                    rotated_shape[i] = (rotated_shape[i][0], -rotated_shape[i][1])


                self.dropped_shapes[self.rot_selected_shape_index] = (rotated_shape, color)

                # Update previous angle for the next frame
                self.prev_angle = angle


            else:
                self.rot_selected_shape_index = self.select_shape_at(px, py)
                print("ROTATION SHAPE INDEX", self.rot_selected_shape_index)
                if self.rot_selected_shape_index is not None:
                    self.prev_angle = angle

        else:
            # Reset rotation state when the gesture is no longer active
            self.rot_selected_shape_index = None
            self.prev_angle = None



if __name__ == '__main__':
    app = DrawingApp()
    app.run()

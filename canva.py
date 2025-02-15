import cv2
import numpy as np
from HandTrackingModule import HandDetector
from shapes import Draw
from clipping import clip_polygon
import math
from transform import Transformation
import copy

class DrawingApp:
    def __init__(self):
        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        self.w, self.h = 1280, 720
        self.cap.set(3, self.w)
        self.cap.set(4, self.h)
        self.eraser_button_clicked = False
        
        self.vertex_selection_radius = 10  # Radius to detect vertex clicks
        self.selected_vertex_index = None  # Track which vertex is selected
        self.selected_vertex_shape = None  # Track which shape the selected vertex belongs to
        
        
        # Load Background Image
        self.background_img_unchanged = cv2.imread("image/image.png")
        self.background_img = copy.deepcopy(self.background_img_unchanged)
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
        self.initial_rectangle_coordinates = [(-70, -40), (70, -40), (70, 40), (-70, 40)]

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

        self.brush_thickness = 4

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
            "thickness line" : [(1110, 330), (1110, 450)],
            "thickness button" : [(1070,100 ), (1150, 180)]
        }
    
    def select_vertex_at(self, x, y):
        """Check if a vertex is clicked and return (shape_index, vertex_index)."""
        for shape_idx, shape in enumerate(self.dropped_shapes):
            if isinstance(shape, tuple):  # Skip circles
                continue
            
            for vertex_idx, (vx, vy) in enumerate(shape):
                # Check if point is within vertex selection radius
                if (x - vx)**2 + (y - vy)**2 <= self.vertex_selection_radius**2:
                    print(f"Vertex {vertex_idx} of shape {shape_idx} selected")
                    return shape_idx, vertex_idx
        return None, None
    
    
    def handle_vertex_dragging(self, hand):
        """Handle vertex dragging for true geometric shearing while preserving parallelism."""
        if self.rotation or self.scaling:
            return

        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.is_index_up(hand):
            if self.selected_vertex_shape is None:
                shape_idx, vertex_idx = self.select_vertex_at(x, y)
                if shape_idx is not None:
                    self.selected_vertex_shape = shape_idx
                    self.selected_vertex_index = vertex_idx
            else:
                shape = self.dropped_shapes[self.selected_vertex_shape]
                if not isinstance(shape, tuple) and self.check_hand_inside_canvas(x, y):
                    shape_list = list(shape)
                    num_vertices = len(shape_list)
                    
                    # Get selected vertex and its initial position
                    orig_x, orig_y = shape_list[self.selected_vertex_index]
                    
                    # Determine the base edge (the edge that remains fixed during shear)
                    next_vertex = (self.selected_vertex_index + 1) % num_vertices
                    prev_vertex = (self.selected_vertex_index - 1) % num_vertices
                    
                    next_x, next_y = shape_list[next_vertex]
                    prev_x, prev_y = shape_list[prev_vertex]
                    
                    # Determine if this is a vertical or horizontal shear
                    is_vertical = abs(orig_x - next_x) < 10 or abs(orig_x - prev_x) < 10
                    
                    # Calculate shear factor and apply transformation
                    if is_vertical:
                        # Vertical shear: keep x-coordinates of the base edge fixed
                        base_x = prev_x if abs(orig_x - prev_x) < 10 else next_x
                        shear_factor = (x - orig_x) / (self.h / 2)  # Normalize by canvas height
                        
                        # Apply vertical shear transformation
                        for i in range(num_vertices):
                            px, py = shape_list[i]
                            if i == self.selected_vertex_index:
                                shape_list[i] = (x, y)
                            elif abs(px - base_x) > 10:  # Don't move vertices on the base edge
                                # Shear transformation: x' = x + shear_factor * (py - orig_y)
                                new_x = px + shear_factor * (py - orig_y)
                                shape_list[i] = (new_x, py)
                    else:
                        # Horizontal shear: keep y-coordinates of the base edge fixed
                        base_y = prev_y if abs(orig_y - prev_y) < 10 else next_y
                        shear_factor = (y - orig_y) / (self.w / 2)  # Normalize by canvas width
                        
                        # Apply horizontal shear transformation
                        for i in range(num_vertices):
                            px, py = shape_list[i]
                            if i == self.selected_vertex_index:
                                shape_list[i] = (x, y)
                            elif abs(py - base_y) > 10:  # Don't move vertices on the base edge
                                # Shear transformation: y' = y + shear_factor * (px - orig_x)
                                new_y = py + shear_factor * (px - orig_x)
                                shape_list[i] = (px, new_y)
                    
                    self.dropped_shapes[self.selected_vertex_shape] = shape_list
        else:
            self.selected_vertex_shape = None
            self.selected_vertex_index = None
            
            
    # ---------------------- Main Loop ----------------------
    def run(self):
        """Main loop for the application."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.h, self.w, _ = frame.shape

            # Define drawing area and background
            drawing_canvas = [(30, 150), (1030, 150), (1030, self.h - 50), (30, self.h - 50)]
            self.background_img = copy.deepcopy( self.background_img_unchanged )
            background_resized = cv2.resize(self.background_img, (self.w, self.h))
            
            # Initialize canvas and save initial state
            if self.canvas is None:
                self.canvas = np.zeros_like(background_resized)

            # Detect hands
            hands, frame = self.detector.findHands(frame, img_to_draw=background_resized)
        
            if hands:
                drawing_hand = None
                helping_hand = None

                # Determine which hand is which
                if len(hands) == 1:
                    drawing_hand = hands[0]
                elif len(hands) == 2:
                    for hand in hands:
                        if hand["type"] == "Left":
                            drawing_hand = hand
                        else:
                            helping_hand = hand
                
                # Handle two-hand gestures
                if drawing_hand and helping_hand:
                    self.handle_reshaping(drawing_hand, helping_hand)
                    self.handle_rotating(drawing_hand, helping_hand)
                
                # Handle single-hand gestures
                if drawing_hand:

                    self.control_line_thickness(img = background_resized, right_hand = drawing_hand, left_hand= helping_hand)
                    self.process_hand_buttons(drawing_hand)
                    self.handle_dragging(drawing_hand)


            
                
                

                    if not self.process_hand_buttons(drawing_hand):
                        self.handle_vertex_dragging(drawing_hand)  # Try vertex dragging first
                        # self.handle_dragging(drawing_hand)
                        if self.selected_vertex_shape is None:
                            self.handle_dragging(drawing_hand)
                        
                    # self.process_hand_buttons(drawing_hand)
                    # self.handle_dragging(drawing_hand)


            # Merge canvas and render shapes
            combined = self.render_canvas(background_resized, drawing_canvas)
            cv2.imshow("Virtual Canvas", combined)
            self.rotation = False

            # Check for quit command
            if cv2.waitKey(1) == ord('q'):
                break

        # Cleanup

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


    def control_line_thickness(self, img, right_hand, left_hand):
        px, py = right_hand["lmList"][8][:2]
        rect = self.buttons["thickness button"]

        if self.is_index_up(right_hand) and self.point_in_rect(rect=rect, x=px, y=py):
            # If left hand is available and shows the proper gesture, update thickness first.
            if left_hand and self.is_index_and_thump_up(left_hand):
                index_tip = np.array(left_hand["lmList"][8][:2])
                thumb_tip = np.array(left_hand["lmList"][4][:2])
                current_distance = np.linalg.norm(index_tip - thumb_tip)
                # For normalization, using a distance between two other landmarks:
                point2 = np.array(left_hand["lmList"][9][:2])
                point1 = np.array(left_hand["lmList"][0][:2])
                current_distance_norm = current_distance / np.linalg.norm(point1 - point2)
                self.brush_thickness = np.interp(current_distance_norm, [0.14, 1.587], [1, 20])
            
            # Now draw the UI indicator on the image that will be rendered.
            line = self.buttons["thickness line"]
            self.draw_shapes.line(img=img, point1=line[0], point2=line[1], color=(255, 0, 255), thickness=3)
            tool_position = np.interp(self.brush_thickness, [1, 20], [line[1][1], line[0][1]])
            cv2.circle(img=img, center=(line[0][0], int(tool_position)), radius=5, color=(255, 0, 255))

            
        


    def render_canvas(self, background, drawing_canvas):
        combined = cv2.addWeighted(background, 0.8, self.canvas, 1, 0)
        for shape in self.dropped_shapes:
            if shape:
                if isinstance(shape, tuple):
                    # Handle circle drawing
                    cx, cy, r = shape
                    cv2.circle(combined, (int(cx), int(cy)), int(r), (0, 0, 255), 2)
                else:
                    # Handle polygon shapes
                    clipped = clip_polygon(shape, drawing_canvas)
                    if clipped:
                        self.draw_shapes.polygon(combined, points=clipped, color=(0, 0, 255))
        return combined

    def select_shape_at(self, x, y):
        """Return the index of a shape that contains point (x,y), or None."""
        for i in range(len(self.dropped_shapes) - 1, -1, -1):
            shape = self.dropped_shapes[i]
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
        
        # Check if clicking the shape button
        if self.point_in_rect(x, y, rect):
            if self.is_index_up(hand):
                self.shape_flags[tool] = True
                self.brush_button_clicked = False
                self.eraser_button_clicked = False
                print(f"{tool.capitalize()} Button Clicked")
                
        # Handle shape dropping
        elif self.shape_flags.get(tool, False) and self.is_index_up(hand):
            # Only drop shape if inside canvas area
            if self.check_hand_inside_canvas(x, y):
                # Create and add the new shape
                self.dropped_shapes.append(drop_func(x, y))
                self.shape_flags[tool] = False
                print(f"{tool.capitalize()} Dropped")
            else:
                # Reset shape flag if dropped outside canvas
                self.shape_flags[tool] = False
                print(f"{tool.capitalize()} Cancelled - Outside Canvas")

    def handle_color_button(self, color, x, y, hand):
        """Generic handler for color buttons."""
        rect = self.buttons[color]
        if self.point_in_rect(x, y, rect) and self.is_index_up(hand):
            for c in self.color_flags:
                self.color_flags[c] = (c == color)
            print(f"{color.capitalize()} Color Clicked")

    def handle_brush(self, x, y, hand):
        """Handle brush and eraser tools with state saving."""
        brush_rect = self.buttons["brush"]
        eraser_rect = self.buttons["eraser"]
        
        # Check for brush button
        if self.point_in_rect(x, y, brush_rect):
            if self.is_index_up(hand):
                self.brush_button_clicked = True
                self.eraser_button_clicked = False
                print("Brush Button Clicked")
                
        # Check for eraser button
        elif self.point_in_rect(x, y, eraser_rect):
            if self.is_index_up(hand):
                self.eraser_button_clicked = True
                self.brush_button_clicked = False
                print("Eraser Button Clicked")
                
        # Handle drawing/erasing
        elif (self.brush_button_clicked or self.eraser_button_clicked) and \
            self.is_index_up(hand) and self.check_hand_inside_canvas(x, y):
            
            # Initialize previous position if not set
            if self.prev_x is None or self.prev_y is None:
                self.prev_x, self.prev_y = x, y

            self.draw_shapes.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color=self.get_color(), thickness= self.brush_thickness)

                return
            
            # Draw line based on selected tool
            if self.brush_button_clicked:
                cv2.line(self.canvas, 
                        (self.prev_x, self.prev_y), 
                        (x, y), 
                        self.get_color(), 
                        thickness=5)
            elif self.eraser_button_clicked:
                cv2.line(self.canvas, 
                        (self.prev_x, self.prev_y), 
                        (x, y), 
                        (0, 0, 0), 
                        thickness=20)
            
            # Update previous position

            self.prev_x, self.prev_y = x, y
        else:
            # Reset previous position when not drawing
            self.prev_x, self.prev_y = None, None

    # ---------------------- Drop Functions for Shapes ----------------------
    def drop_square(self, x, y):
        return [(dx + x, dy + y) for dx, dy in self.initial_square_coordinates]

    def drop_rectangle(self, x, y):
        return [(dx + x, dy + y) for dx, dy in self.initial_rectangle_coordinates]

    def drop_triangle(self, x, y):
        triangle_size = 50
        return [(x, y - triangle_size),
                (x - triangle_size, y + triangle_size),
                (x + triangle_size, y + triangle_size)]

    def drop_line(self, x, y):
        line_length = 80
        return [(x - line_length // 2, y), (x + line_length // 2, y)]

    def drop_circle(self, x, y):
        circle_radius = 40
        return (x, y, circle_radius)

    # ---------------------- Process Hand Buttons ----------------------
    def process_hand_buttons(self, hand):
        """Process all button interactions including undo/redo."""
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

       
        # Process other buttons only if undo/redo wasn't triggered
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
        """Handle shape dragging with state saving."""
        # Skip if rotation is active
        if self.rotation or self.selected_vertex_shape is not None:
            return
        
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.is_index_up(hand):
            if self.selected_shape_index is not None:
                # Initialize previous position if not set
                if self.prev_index_x is None or self.prev_index_y is None:
                    self.prev_index_x, self.prev_index_y = x, y
                    return
                
                # Calculate movement
                dx = x - self.prev_index_x
                dy = y - self.prev_index_y
                
                # Update shape position
                shape = self.dropped_shapes[self.selected_shape_index]
                shape_moved = False
                
                if isinstance(shape, tuple):
                    # Handle circle dragging
                    cx, cy, r = shape
                    self.dropped_shapes[self.selected_shape_index] = (cx + dx, cy + dy, r)
                    shape_moved = True
                else:
                    # Handle polygon dragging
                    self.dropped_shapes[self.selected_shape_index] = [
                        (px + dx, py + dy) for (px, py) in shape
                    ]
                    shape_moved = True
                
                # Update previous position
                self.prev_index_x, self.prev_index_y = x, y
                
                # Save state if shape was moved
                
            else:
                # Try to select a shape under the finger
                self.selected_shape_index = self.select_shape_at(x, y)
                if self.selected_shape_index is not None:
                    self.prev_index_x, self.prev_index_y = x, y
        else:
            # Reset selection when gesture ends
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

                selected_shape = self.original_shape_for_scaling

                # Handle circles differently from polygons
                if isinstance(selected_shape, tuple):
                    # For circles: (cx, cy, r)
                    cx, cy, original_radius = selected_shape
                    new_radius = int(original_radius * scaling_ratio)
                    self.dropped_shapes[self.scaling_selected_shape_index] = (cx, cy, new_radius)
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
                    self.dropped_shapes[self.scaling_selected_shape_index] = scaled_shape
        else:
            # Reset scaling state when gesture ends
            self.scaling_selected_shape_index = None
            self.initial_scaling_distance = None
            self.original_shape_for_scaling = None
            self.scaling = False


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

                selected_shape = self.dropped_shapes[self.rot_selected_shape_index]
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


                self.dropped_shapes[self.rot_selected_shape_index] = rotated_shape

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
import cv2
import numpy as np
from HandTrackingModule import HandDetector
from shapes import Draw
from clipping import clip_polygon
import math
from transform import Transformation
import copy
from RayCasting import is_point_in_polygon
from colorfill import fill_poly

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
        
        self.shape_colors = {}  # Dictionary to store shape colors
        self.currently_filling = False # Flag to track if filling is in progress    
        
        self.edge_selection_radius = 10
        self.selected_edge = None  # Will store (shape_index, start_idx, end_idx)
        self.edge_drag_start = None
            
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
        self.undo_in_process = False
        self.redo_in_process = False
        self.shape_filling = False
        self.edge_extension_running = False
        self.vertex_dragging_running = False

        # Drawing Canvas
        self.drawing_canvas = [(30, 115), (1032, 115), (1032, 695), (30, 695)]

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
        self.brush_drawing = False               # **NEW: flag to detect brush stroke action

        # Variables for dragging/resizing shapes
        self.selected_shape_index = None
        self.prev_index_x, self.prev_index_y = None, None
        self.dragging_active = False             # **NEW: flag for dragging action

        self.prev_angle = None
        self.rotation = False
        self.rot_selected_shape_index = None
        self.scaling_selected_shape_index = None
        self.initial_scaling_distance = None
        self.original_shape_for_scaling = None
        self.scaling = False
        self.scaling_active = False               # **NEW: flag for scaling action
        self.rotating_active = False              # **NEW: flag for rotation action
        self.hover_shape_index = None               
        self.hover_highlight_color = (0, 255, 0)
        self.hover_thickness = 2
        
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
            "undo": [(1070, 20), (1150, 100)],      # **UNDO button area
            "redo": [(1150, 20), (1230, 100)],     # **REDO button area
            "thickness line" : [(1110, 280), (1110, 500)],
            "thickness button" : [(1150, 120), (1240, 200)], 
        }
        
        # **NEW: Initialize undo and redo stacks.
        self.undo_stack = []
        self.redo_stack = []
        
    def detect_hover(self, hand):
        if hand is None:
            self.hover_shape_index = None
            return None
        
        fingers = self.detector.fingersUp(hand)
        if fingers[0] == 1 and fingers[1] == 1 and all(f == 0 for f in fingers[2:]):
            lm_list = hand["lmList"]
            x,y = lm_list[8][:2]
            self.hover_shape_index = self.select_shape_at(x, y)
            
        else:
            self.hover_shape_index = None
        

    # **NEW: Method to record the current drawing state.
    def record_state(self):
        state = {
            'canvas': self.canvas.copy(),
            'dropped_shapes': copy.deepcopy(self.dropped_shapes),
            'shape_colors': copy.deepcopy(self.shape_colors)
        }

        
        self.undo_stack.append(state)
        self.redo_stack.clear()
        print("State recorded. Undo stack size:", len(self.undo_stack))

    # **NEW: Undo method.
    def undo(self, hand):
        """Perform undo if the hand is within the undo button area and gesture is valid."""

        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]
        
        if self.undo_in_process :
            if self.point_in_rect(x, y, self.buttons["undo"]) and self.is_index_up(hand):
                return

       
        if self.point_in_rect(x, y, self.buttons["undo"]) and self.is_index_up(hand):
            self.undo_in_process = True
            if not self.undo_stack:
                print("Nothing to undo")
                return   # Return True so that the main loop can delay.
            # Save current state into redo stack.
            current_state = {
                'canvas': self.canvas.copy(),
                'dropped_shapes': copy.deepcopy(self.dropped_shapes),
                'shape_colors': copy.deepcopy(self.shape_colors)
            }
            self.redo_stack.append(current_state)
            state = self.undo_stack.pop()
            self.canvas = state['canvas'].copy()
            self.dropped_shapes = copy.deepcopy(state['dropped_shapes'])
            self.shape_colors = copy.deepcopy(state['shape_colors'])
            print("Undo performed. Undo stack size:", len(self.undo_stack))
            
        else:
            self.undo_in_process = False

    def redo(self, hand):
        """Perform redo if the hand is within the redo button area and gesture is valid."""
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.redo_in_process :
            if self.point_in_rect(x, y, self.buttons["redo"]) and self.is_index_up(hand):
                return     


        if self.point_in_rect(x, y, self.buttons["redo"]) and self.is_index_up(hand):
            self.redo_in_process = True
            if not self.redo_stack:
                print("Nothing to redo")
                return 
            current_state = {
                'canvas': self.canvas.copy(),
                'dropped_shapes': copy.deepcopy(self.dropped_shapes),
                'shape_colors': copy.deepcopy(self.shape_colors)
            }
            self.undo_stack.append(current_state)
            state = self.redo_stack.pop()
            self.canvas = state['canvas'].copy()
            self.dropped_shapes = copy.deepcopy(state['dropped_shapes'])
            self.shape_colors = copy.deepcopy(state['shape_colors'])
            print("Redo performed. Redo stack size:", len(self.redo_stack))
        
        else:
            self.redo_in_process = False
        

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
    
    def fill_shape(self, x, y, hand):
        """Handle shape filling with selected color."""
        if self.is_index_up(hand):
            shape_idx = self.select_shape_at(x, y)
            if shape_idx is not None:
                if not self.shape_filling: 
                    self.record_state()
                current_color = self.get_color()
                # Update or add color for the shape
                self.shape_colors[shape_idx] = current_color
                print(f"Shape {shape_idx} filled with color {current_color}")
                #self.record_state()  # **Record state after filling.
                self.shape_filling = True
                return True
            
        self.shape_filling = False
        return False
            
    def select_edge_at(self, x, y):
        """Check if point is near an edge and return (shape_idx, start_idx, end_idx)."""
        for shape_idx, shape in enumerate(self.dropped_shapes):
            if isinstance(shape, tuple):  # Skip circles
                continue
            
            for i in range(len(shape)):
                start = shape[i]
                end = shape[(i + 1) % len(shape)]
                
                # Calculate distance from point to line segment
                dist = self.point_to_line_distance(x, y, start, end)
                if dist < self.edge_selection_radius:
                    return shape_idx, i, (i + 1) % len(shape)
        return None

    def point_to_line_distance(self, x, y, start, end):
        """Calculate distance from point to line segment."""
        x1, y1 = start
        x2, y2 = end
        
        # Calculate squared length of line segment
        line_length_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_length_sq == 0:
            return math.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate projection point
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_length_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)

    def handle_edge_extension(self, hand):
        """Handle edge extension while preventing overlap with dragging."""
        # Skip if rotation, scaling or dragging is active
        if self.rotation or self.scaling or self.selected_shape_index is not None:
            return

        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.is_index_up(hand):
            if self.selected_edge is None:
                edge_info = self.select_edge_at(x, y)
                if edge_info:
                    self.selected_edge = edge_info
                    self.edge_drag_start = (x, y)
            else:
                if self.check_hand_inside_canvas(x, y):

                    if not self.edge_extension_running:
                        self.record_state()

                    shape_idx, start_idx, end_idx = self.selected_edge
                    shape = list(self.dropped_shapes[shape_idx])
                    
                    # Get the selected edge vertices
                    start = shape[start_idx]
                    end = shape[end_idx]
                    
                    # Determine if this is a vertical or horizontal edge
                    is_vertical = abs(end[0] - start[0]) < 10
                    
                    if is_vertical:
                        # For vertical edges, only move the x-coordinate
                        target_x = x
                        # Find all vertices with same x-coordinate
                        ref_x = start[0]
                        for i in range(len(shape)):
                            if abs(shape[i][0] - ref_x) < 10:
                                shape[i] = (target_x, shape[i][1])
                    else:
                        # For horizontal edges, only move the y-coordinate
                        target_y = y
                        # Find all vertices with same y-coordinate
                        ref_y = start[1]
                        for i in range(len(shape)):
                            if abs(shape[i][1] - ref_y) < 10:
                                shape[i] = (shape[i][0], target_y)
                    
                    self.edge_extension_running = True                    
                    self.dropped_shapes[shape_idx] = shape
        else:
            self.selected_edge = None
            self.edge_drag_start = None
            self.edge_extension_running = False
        
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

                if not self.vertex_dragging_running:
                    self.record_state() 

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
                    self.vertex_dragging_running = True
        else:
            self.vertex_dragging_running = False
            self.selected_vertex_shape = None
            self.selected_vertex_index = None
            
    def run(self):
        """Main loop for the application."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self.h, self.w, _ = frame.shape

            # Define drawing area and background
            self.background_img = copy.deepcopy(self.background_img_unchanged)
            background_resized = cv2.resize(self.background_img, (self.w, self.h))
            
            # Initialize canvas and save initial state
            if self.canvas is None:
                self.canvas = np.zeros_like(background_resized)
                # Record the initial (empty) state.
                self.record_state()

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
                
                self.detect_hover(drawing_hand)
                # Handle two-hand gestures
                if drawing_hand and helping_hand:
                    self.handle_reshaping(drawing_hand, helping_hand)
                    self.handle_rotating(drawing_hand, helping_hand)
                
                # Handle single-hand gestures
                if drawing_hand:
                    self.control_line_thickness(img=background_resized, right_hand=drawing_hand, left_hand=helping_hand)
                    # **NEW: Process undo/redo first.
                    lm_list = drawing_hand["lmList"]
                    x, y = lm_list[8][:2]
      
         

                    self.undo(drawing_hand)
                    self.redo(drawing_hand)
                       

                    self.process_hand_buttons(drawing_hand)
                    self.handle_dragging(drawing_hand)
                    
                    # If not processing a button, allow edge extension (or dragging if no edge is selected)
                    if not self.process_hand_buttons(drawing_hand):
                        self.handle_edge_extension(drawing_hand)
                        if self.selected_edge is None:
                            self.handle_dragging(drawing_hand)

            # Merge canvas and render shapes
            combined = self.render_canvas(background_resized, self.drawing_canvas)
            cv2.imshow("Virtual Canvas", combined)
            self.rotation = False

            # Check for quit command
            if cv2.waitKey(1) == ord('q'):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

    def point_in_rect(self, x, y, rect):
        (x1, y1), (x2, y2) = rect
        return x1 < x < x2 and y1 < y < y2

    def check_hand_inside_canvas(self, x, y):
        return 30 < x < 1032 and 115 < y < 695

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
                print(current_distance_norm)
                self.brush_thickness = np.interp(current_distance_norm, [0.14, 1.8], [1, 100])
            
            # Now draw the UI indicator on the image that will be rendered.
            line = self.buttons["thickness line"]
            self.draw_shapes.line(img=img, point1=line[0], point2=line[1], color=(255, 0, 255), thickness=3)
            tool_position = np.interp(self.brush_thickness, [1, 100], [line[1][1], line[0][1]])
            self.draw_shapes.draw_circle(img, center=(line[0][0], int(tool_position)), radius=5, 
                             outline_color=(0,0,0),thickness= 1,fill_color= (255, 0, 255) )

    # Modify render_canvas method to use the new circle drawing function:
    def render_canvas(self, background, drawing_canvas):
        # Merge the background and persistent canvas.
        combined = cv2.addWeighted(background, 0.8, self.canvas, 1, 0)
        
        for idx, shape in enumerate(self.dropped_shapes):
            if shape:
                is_highlighted = (idx == self.hover_shape_index)
                outline_color = (0, 0, 255)  # Default outline color
                
                # Get the fill color for this shape if it was set.
                fill_color = self.shape_colors.get(idx, None)
                
                if isinstance(shape, tuple):
                    # Handle circle shapes: shape = (cx, cy, r)
                    cx, cy, r = shape
                    # Fill the circle if a fill color exists.
                    
                    # Draw the circle outline.
                    self.draw_shapes.draw_circle(
                        combined, 
                        center= (int(cx), int(cy)), radius= int(r), 
                        outline_color= outline_color,
                        thickness=2 if is_highlighted else 1,
                        is_highlighted=is_highlighted,
                        fill_color= fill_color
                    )
                else:
                    # Handle polygon shapes.
                    clipped = clip_polygon(shape, drawing_canvas)
                    if clipped:
                        pts = np.array(clipped, np.int32)
                        # Fill the polygon if a fill color exists.
                        if fill_color is not None:
                            fill_poly(combined, pts, fill_color)
                        # Draw the polygon outline.
                        if is_highlighted:
                            self.draw_shapes.polygon(combined, pts, (180, 180, 180), thickness=2)
                            self.draw_shapes.polygon(combined, pts, outline_color, thickness=2)
                        else:
                            self.draw_shapes.polygon(combined, pts, outline_color, thickness=1)
        
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
                if is_point_in_polygon((x, y), pts):
                    print(f"Shape {i} selected for dragging")
                    return i
        return None

    def is_point_in_circle(self, x, y, circle):
        cx, cy, r = circle
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    # ---------------------- Generic Tool Handlers ----------------------
    def handle_dragging(self, hand):
        """Handle shape dragging."""
        # Skip if rotation or edge extension is active
        if self.rotation or self.selected_edge is not None:
            return
        
        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.is_index_up(hand):
            if self.selected_shape_index is not None:

                if not self.dragging_active:
                    self.record_state()  # **Record state after dragging finishes.
                    

                # Initialize previous position if not set
                if self.prev_index_x is None or self.prev_index_y is None:
                    self.prev_index_x, self.prev_index_y = x, y
                    self.dragging_active = True  # **Start dragging action
                    return
                self.dragging_active = True
                # Calculate movement
                dx = x - self.prev_index_x
                dy = y - self.prev_index_y
                
                # Update shape position
                shape = self.dropped_shapes[self.selected_shape_index]
                
                if isinstance(shape, tuple):
                    # Handle circle dragging
                    cx, cy, r = shape
                    new_cx = cx + dx
                    new_cy = cy + dy
                    # Check if new position is within canvas
                    if self.check_hand_inside_canvas(new_cx, new_cy):
                        self.dropped_shapes[self.selected_shape_index] = (new_cx, new_cy, r)
                else:
                    # Handle polygon dragging
                    new_shape = []
                    for px, py in shape:
                        new_x = px + dx
                        new_y = py + dy
                        new_shape.append((new_x, new_y))
                    
                    # Clip the shape to the canvas
                    clipped_shape = clip_polygon(new_shape, self.drawing_canvas)
                    if clipped_shape:
                        self.dropped_shapes[self.selected_shape_index] = new_shape if self.check_hand_inside_canvas(x, y) else clipped_shape
                
                # Update previous position
                self.prev_index_x, self.prev_index_y = x, y
                
            else:
                # Try to select a shape under the finger
                self.selected_shape_index = self.select_shape_at(x, y)
                if self.selected_shape_index is not None:
                    self.prev_index_x, self.prev_index_y = x, y
                self.dragging_active = False
        else:
            self.dragging_active = False
            self.selected_shape_index = None
            self.prev_index_x, self.prev_index_y = None, None
        
    def handle_shape_button(self, tool, drop_func, x, y, hand):
        """Generic handler for shape buttons (square, rectangle, etc.)."""
        rect = self.buttons[tool]
        
        # Check if clicking the shape button
        if self.point_in_rect(x, y, rect):
            if self.is_index_up(hand):
                self.shape_flags[tool] = True
                self.brush_button_clicked = False
                self.eraser_button_clicked = False
                # Reset any color selection when selecting a shape tool
                for color in self.color_flags:
                    self.color_flags[color] = False
                print(f"{tool.capitalize()} Button Clicked")
                
        # Handle shape dropping
        elif self.shape_flags.get(tool, False) and self.is_index_up(hand):
            # Only drop shape if inside canvas area
            if self.check_hand_inside_canvas(x, y):
                # Add new shape without any color association
                self.record_state() # record before dropping a shape
                new_shape = drop_func(x, y)
                self.dropped_shapes.append(new_shape)
                # Reset shape flag
                self.shape_flags[tool] = False
                print(f"{tool.capitalize()} Dropped")
                #self.record_state()  # **Record state after dropping a shape.
            else:
                # Reset shape flag if dropped outside canvas
                self.shape_flags[tool] = False
                print(f"{tool.capitalize()} Cancelled - Outside Canvas")

    def handle_color_button(self, color, x, y, hand):
        """Generic handler for color buttons."""
        rect = self.buttons[color]
        if self.point_in_rect(x, y, rect) and self.is_index_up(hand):
            # Only set the selected color flag, don't auto-fill shapes
            for c in self.color_flags:
                self.color_flags[c] = (c == color)
            print(f"{color.capitalize()} Color Selected")

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
            
            if not self.brush_drawing:
                self.record_state()

            # Initialize previous position if not set
            if self.prev_x is None or self.prev_y is None:
                self.prev_x, self.prev_y = x, y
                self.brush_drawing = True  # **Start brush drawing action

            self.draw_shapes.line(self.canvas, (self.prev_x, self.prev_y), (x, y), color=self.get_color(), thickness= int(self.brush_thickness))

            # Draw line based on selected tool
            if self.brush_button_clicked:
                self.draw_shapes.line(self.canvas, 
                        (self.prev_x, self.prev_y), 
                        (x, y), 
                        self.get_color(), 
                        thickness= int(self.brush_thickness))
            elif self.eraser_button_clicked:
                self.draw_shapes.line(self.canvas, 
                        (self.prev_x, self.prev_y), 
                        (x, y), 
                        (0, 0, 0), 
                        thickness = int(self.brush_thickness))
            
            # Update previous position
            self.prev_x, self.prev_y = x, y
        else:
            # If drawing was in progress and now finished, record state.
            # if self.brush_drawing:
            #     self.record_state()
            #     self.brush_drawing = False
            self.brush_drawing = False
            self.prev_x, self.prev_y = None, None

    # ---------------------- Drop Functions for Shapes ----------------------
    def drop_rectangle(self, x, y):
        # Return shape without any color fill
        return [(dx + x, dy + y) for dx, dy in self.initial_rectangle_coordinates]

    def drop_square(self, x, y):
        # Return shape without any color fill
        return [(dx + x, dy + y) for dx, dy in self.initial_square_coordinates]

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
        
        # Process fill if a color is selected
        if self.check_hand_inside_canvas(x, y):
            if any(self.color_flags.values()):
                if self.fill_shape(x, y, hand):
                    return True

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
        
        return False

    def handle_edge_extension(self, hand):
        """Handle edge extension while preventing overlap with dragging."""
        # Skip if rotation, scaling or dragging is active
        if self.rotation or self.scaling or self.selected_shape_index is not None:
            return

        lm_list = hand["lmList"]
        x, y = lm_list[8][:2]

        if self.is_index_up(hand):
            if self.selected_edge is None:
                edge_info = self.select_edge_at(x, y)
                if edge_info:
                    self.selected_edge = edge_info
                    self.edge_drag_start = (x, y)
            else:
                if self.check_hand_inside_canvas(x, y):
                    shape_idx, start_idx, end_idx = self.selected_edge
                    shape = list(self.dropped_shapes[shape_idx])
                    
                    # Get the selected edge vertices
                    start = shape[start_idx]
                    end = shape[end_idx]
                    
                    # Determine if this is a vertical or horizontal edge
                    is_vertical = abs(end[0] - start[0]) < 10
                    
                    if is_vertical:
                        # For vertical edges, only move the x-coordinate
                        target_x = x
                        # Find all vertices with same x-coordinate
                        ref_x = start[0]
                        for i in range(len(shape)):
                            if abs(shape[i][0] - ref_x) < 10:
                                shape[i] = (target_x, shape[i][1])
                    else:
                        # For horizontal edges, only move the y-coordinate
                        target_y = y
                        # Find all vertices with same y-coordinate
                        ref_y = start[1]
                        for i in range(len(shape)):
                            if abs(shape[i][1] - ref_y) < 10:
                                shape[i] = (shape[i][0], target_y)
                    
                    self.dropped_shapes[shape_idx] = shape
        else:
            self.selected_edge = None
            self.edge_drag_start = None
    
    def is_index_and_thump_up(self, hand):
        fingers = self.detector.fingersUp(hand)
        return fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0

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

                if not self.scaling_active:
                    self.record_state()

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
                    pts = np.array(selected_shape)
                    pivx, pivy = np.mean(pts, axis=0)
                    scaled_shape = self.transformation.scale(
                        points=selected_shape,
                        pivot=(pivx, pivy),
                        sx=scaling_ratio,
                        sy=scaling_ratio
                    )
                    self.dropped_shapes[self.scaling_selected_shape_index] = scaled_shape
                self.scaling_active = True  # **Scaling in progress
        else:
            # if self.scaling_active:
            #     self.record_state()  # **Record state after scaling completes.
            #     self.scaling_active = False
            self.scaling_active = False
            self.scaling_selected_shape_index = None
            self.initial_scaling_distance = None
            self.original_shape_for_scaling = None
            self.scaling = False

    def is_hand_closed(self, hand):
        fingers_up = self.detector.fingersUp(hand)
        return sum(fingers_up) == 0
    
    def index_and_middle_up(self, hand):
        fingers_up = self.detector.fingersUp(hand)
        return fingers_up[0] == 0 and fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 0 and fingers_up[4] == 0
    
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

                if not self.rotating_active:
                    self.record_state()


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
                for x_val, y_val in selected_shape:
                    pivx += x_val
                    pivy += y_val
                pivx /= len(selected_shape)
                pivy /= len(selected_shape)

                # Rotate the shape by the computed degree difference times sensitivity
                rotated_shape = self.transformation.rotate(selected_shape, (pivx, pivy), dtheta_deg * sensitivity)

                # Update the shape with its rotated coordinates
                for i in range(len(rotated_shape)):
                    rotated_shape[i] = (rotated_shape[i][0], -rotated_shape[i][1])
                self.dropped_shapes[self.rot_selected_shape_index] = rotated_shape
                self.rotating_active = True  # **Rotation in progress
                self.prev_angle = angle
            else:
                self.rot_selected_shape_index = self.select_shape_at(px, py)
                print("ROTATION SHAPE INDEX", self.rot_selected_shape_index)
                if self.rot_selected_shape_index is not None:
                    self.prev_angle = angle
        else:
            # if self.rotating_active:
            #     self.record_state()  # **Record state after rotation completes.
            #     self.rotating_active = False
            self.rotating_active = False
            self.rot_selected_shape_index = None
            self.prev_angle = None
        
if __name__ == '__main__':
    app = DrawingApp()
    app.run()

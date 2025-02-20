import math
import numpy as np

def fill_poly(canvas, pts, fill_color):
    """
    Fills a polygon on the given canvas with fill_color using a scan-line fill algorithm.
    
    Args:
        canvas (numpy.ndarray): The image (height x width x channels) to fill.
        pts (list of tuple): List of (x, y) vertices defining the polygon.
        fill_color (tuple): Color to fill the polygon (B, G, R).
    """
    # Convert vertices to integer coordinates.
    pts_int = [(int(round(x)), int(round(y))) for (x, y) in pts]
    
    # Compute the bounding box of the polygon.
    min_y = min(y for (_, y) in pts_int)
    max_y = max(y for (_, y) in pts_int)
    
    height, width = canvas.shape[:2]
    
    # Process each horizontal scanline in the bounding box.
    for y in range(min_y, max_y + 1):
        intersections = []
        n = len(pts_int)
        for i in range(n):
            p1 = pts_int[i]
            p2 = pts_int[(i + 1) % n]  # Next vertex (with wrap-around)
            
            # Order the points so that p1 is the lower point.
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
            
            # Check if the scanline crosses the edge.
            # Use the rule: include the lower endpoint and exclude the upper endpoint.
            if p1[1] <= y < p2[1]:
                # Avoid division by zero for horizontal segments.
                if p2[1] - p1[1] != 0:
                    x_int = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / float(p2[1] - p1[1])
                    intersections.append(x_int)
        
        # Sort the intersections along the x-axis.
        intersections.sort()
        
        # Fill between pairs of intersections.
        for j in range(0, len(intersections), 2):
            if j + 1 < len(intersections):
                x_start = int(math.ceil(intersections[j]))
                x_end = int(math.floor(intersections[j + 1]))
                # Clamp x values to the canvas boundaries.
                x_start = max(x_start, 0)
                x_end = min(x_end, width - 1)
                for x in range(x_start, x_end + 1):
                    if 0 <= y < height:
                        canvas[y, x] = fill_color

# In your render_canvas function, update the call to fill_poly:

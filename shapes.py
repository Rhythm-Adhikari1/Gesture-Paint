import cv2
import math
import numpy as np

class Draw:

    def __init__(self):
        pass

    def line(self, img, point1, point2, color, thickness=1):
        x1, y1 = map(int, point1)  # Ensure integer values
        x2, y2 = map(int, point2)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x2 > x1 else -1  
        sy = 1 if y2 > y1 else -1  
        err = dx - dy  
        
        while True:
            # Draw the pixel (or a small circle for thickness > 1)
            if thickness == 1:
                img[y1, x1] = color 
            else:
                # Draw a filled circle with radius thickness//2
                self.draw_circle(img, center=(x1, y1), radius=int(thickness // 2), 
                            outline_color=color, thickness=-1, fill_color= color )

            
            # Check if reached the end point
            if x1 == x2 and y1 == y2:  
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return img

    def polygon(self, img, points, color, thickness=1):
     
        n = len(points)
        if n < 2:
            return  # No polygon to draw

        # Convert points to NumPy array with the required shape (n, 1, 2)
        pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
         # Draw lines manually to form the polygon
        for i in range(n):
            start = tuple(pts_array[i][0])
            end = tuple(pts_array[(i + 1) % n][0])  # Wrap around to close polygon
            self.line(img, start, end, color=color, thickness=thickness)

    
    def draw_circle(self, img, center, radius, outline_color, thickness=1, 
                    is_highlighted=False, fill_color=None):
       
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = int(round(radius))
        
        # Fill the circle using a scanline fill algorithm.
        if fill_color is not None:
            self.fill_circle(img, (cx, cy), r, fill_color)
        
        # If highlighting is enabled, draw a shadow (with a slightly larger radius).
        if is_highlighted:
            self._draw_circle_outline(img, cx, cy, r + 2, (180, 180, 180), thickness)
        
        # Draw the main circle outline using the midpoint algorithm.
        self._draw_circle_outline(img, cx, cy, r, outline_color, max(1, thickness))


    def _draw_circle_outline(self, img, cx, cy, r, color, thickness=1):
        
        def plot_circle_points(x, y):
            # Return symmetric points in all eight octants.
            return [
                (cx + x, cy + y), (cx - x, cy + y),
                (cx + x, cy - y), (cx - x, cy - y),
                (cx + y, cy + x), (cx - y, cy + x),
                (cx + y, cy - x), (cx - y, cy - x)
            ]
        
        x, y = 0, r
        p = 1 - r
        prev_points = None
        
        while x <= y:
            points = plot_circle_points(x, y)
            if prev_points is not None:
                # Draw lines connecting previous and current points for smoothness.
                for i in range(len(points)):
                    pt1 = (int(round(prev_points[i][0])), int(round(prev_points[i][1])))
                    pt2 = (int(round(points[i][0])), int(round(points[i][1])))
                    cv2.line(img, pt1, pt2, color, thickness)
            prev_points = points
            
            if p <= 0:
                x += 1
                p += 2 * x + 1
            else:
                x += 1
                y -= 1
                p += 2 * (x - y) + 1

    def fill_circle(self, img, center, radius, fill_color):
       
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = int(round(radius))
        height, width = img.shape[:2]
        
        # Loop over each scanline that might intersect the circle.
        for y in range(cy - r, cy + r + 1):
            if y < 0 or y >= height:
                continue
            # Calculate horizontal span for the current scanline using circle equation.
            dx = int(math.sqrt(r * r - (y - cy) * (y - cy)))
            x_start = max(cx - dx, 0)
            x_end = min(cx + dx, width - 1)
            for x in range(x_start, x_end + 1):
                img[y, x] = fill_color


def main():
    pass

if __name__ == "__main__":
    main()
            

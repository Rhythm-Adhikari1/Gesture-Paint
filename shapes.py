import cv2

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
                cv2.circle(img, (x1, y1), int(thickness // 2), color, -1)
            
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

    def polygon(self, img, points, color):
        n = len(points)
    
        for i in range(n):
            if i < (n-1):
                self.line(img, points[i], points[i+1],color=color)
            else:
                self.line(img, points[i], points[0], color=color)
    
    
    
    def draw_circle(self, combined, cx, cy, r, color, thickness=1, is_highlighted=False):
        """Draw circle using midpoint circle algorithm."""
        def plot_circle_points(x, y, cx, cy):
            # Plot points in all octants
            points = [
                (cx + x, cy + y), (cx - x, cy + y),
                (cx + x, cy - y), (cx - x, cy - y),
                (cx + y, cy + x), (cx - y, cy + x),
                (cx + y, cy - x), (cx - y, cy - x)
            ]
            return points

    # If highlighted, draw shadow first
        if is_highlighted:
            r_shadow = r + 2
            # Draw shadow using midpoint algorithm
            x, y = 0, r_shadow
            p = 1 - r_shadow
            prev_points = None
            
            while x <= y:
                points = plot_circle_points(x, y, cx, cy)
                if prev_points:
                    for i in range(len(points)):
                        cv2.line(combined, 
                                (int(prev_points[i][0]), int(prev_points[i][1])), 
                                (int(points[i][0]), int(points[i][1])), 
                                (180, 180, 180), 2)
                prev_points = points
                
                if p <= 0:
                    x += 1
                    p = p + 2 * x + 1
                else:
                    x += 1
                    y -= 1
                    p = p + 2 * (x - y) + 1

        # Draw main circle
        x, y = 0, r
        p = 1 - r
        prev_points = None
        
        while x <= y:
            points = plot_circle_points(x, y, cx, cy)
            if prev_points:
                for i in range(len(points)):
                    cv2.line(combined, 
                            (int(prev_points[i][0]), int(prev_points[i][1])), 
                            (int(points[i][0]), int(points[i][1])), 
                            color, thickness)
            prev_points = points
            
            if p <= 0:
                x += 1
                p = p + 2 * x + 1
            else:
                x += 1
                y -= 1
                p = p + 2 * (x - y) + 1

def main():
    pass

if __name__ == "__main__":
    main()
            

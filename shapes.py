import  numpy as np
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
    


def main():
    pass

if __name__ == "__main__":
    main()
            

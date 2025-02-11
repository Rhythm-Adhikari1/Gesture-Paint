import numpy as np


class Transformation:

    def __init__(self):
        pass

    @staticmethod
    def rotate(points, pivot, angle):

        cx, cy = pivot
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        transformed_points = []
        for x, y in points:
            
            x -= cx
            y -= cy

            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a

            x_new += cx
            y_new += cy

            transformed_points.append((int(x_new), int(y_new)))

        return transformed_points

    @staticmethod
    def scale(points, pivot, sx, sy):
      
        cx, cy = pivot

        transformed_points = []
        for x, y in points:
            # Translate to origin
            x -= cx
            y -= cy

            # Apply scaling
            x_new = x * sx
            y_new = y * sy

            # Translate back
            x_new += cx
            y_new += cy

            transformed_points.append((int(x_new), int(y_new)))

        return transformed_points

    @staticmethod
    def shear(points, pivot, shx=0, shy=0):
        
        cx, cy = pivot

        transformed_points = []
        for x, y in points:
            # Translate to origin
            x -= cx
            y -= cy

            # Apply shearing
            x_new = x + shx * y
            y_new = y + shy * x

            # Translate back
            x_new += cx
            y_new += cy

            transformed_points.append((int(x_new), int(y_new)))

        return transformed_points


def main():
    pass

if __name__ == "__main__":
    main()
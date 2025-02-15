import numpy as np


class Transformation:

    def __init__(self):
        pass

    @staticmethod

    def rotate(points, pivot, angle_deg):
        """
        Rotates a list of points around a pivot by a given angle (in degrees)
        using vectorized NumPy operations.

        :param points: List of (x, y) tuples representing the shape's vertices.
        :param pivot: Tuple (px, py) representing the pivot point (e.g., centroid).
        :param angle_deg: Rotation angle in degrees.
        :return: List of rotated points (as tuples of integers).
        """
        # Convert the angle to radians
        angle_rad = np.radians(angle_deg)
        # Build the rotation matrix
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        
        # Convert points and pivot to NumPy arrays
        pts = np.array(points)  # shape: (N, 2)
        pivot = np.array(pivot)  # shape: (2,)
        
        # Translate points so that pivot is at the origin, rotate, then translate back
        rotated_pts = (pts - pivot) @ R.T + pivot
        
        # Convert the results back to integer tuples (if needed)
        return [tuple(np.round(p).astype(int)) for p in rotated_pts]

    @staticmethod
    def scale( points, pivot, sx, sy):
        scaled_points = []
        for (x, y) in points:
            new_x = pivot[0] + sx * (x - pivot[0])
            new_y = pivot[1] + sy * (y - pivot[1])
            scaled_points.append((new_x, new_y))
        return scaled_points

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
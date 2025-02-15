import math
import numpy as np
import cv2

class Fill:
  def scanline_fill(self,  polygon, color):
        """
        Implements the Scanline Fill Algorithm to fill a polygon.
        
        :param polygon: List of (x, y) tuples representing the polygon vertices.
        :param color: Tuple (B, G, R) for the fill color.
        """
        if not polygon:
            return

        # Convert to numpy array
        polygon = np.array(polygon, np.int32)

        # Get min and max Y-values
        ymin = min(polygon[:, 1])
        ymax = max(polygon[:, 1])

        # Edge Table (ET)
        edge_table = [[] for _ in range(ymax - ymin + 1)]

        # Populate Edge Table
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]

            if y1 == y2:  # Ignore horizontal edges
                continue
            
            if y1 > y2:  # Ensure y1 is always the smaller y
                x1, x2 = x2, x1
                y1, y2 = y2, y1

            inverse_slope = (x2 - x1) / (y2 - y1)  # dx/dy
            edge_table[y1 - ymin].append([x1, y2, inverse_slope])  # Store [x, ymax, slope]

        # Active Edge Table (AET)
        active_edges = []

        # Start filling from ymin to ymax
        for y in range(ymin, ymax):
            # Add edges from ET to AET
            active_edges.extend(edge_table[y - ymin])
            active_edges = [e for e in active_edges if e[1] > y]  # Remove edges where y == ymax

            # Sort by x-coordinate
            active_edges.sort()

            # Fill between pairs of active edges
            for i in range(0, len(active_edges), 2):
                x_start = math.ceil(active_edges[i][0])
                x_end = math.floor(active_edges[i + 1][0])
                
                for x in range(x_start, x_end + 1):
                    cv2.circle(self.canvas, (x, y), 1, color, -1)  # Fill pixel

            # Update X-coordinates using inverse slopes
            for edge in active_edges:
                edge[0] += edge[2]  # x += slope

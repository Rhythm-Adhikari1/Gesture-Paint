def is_point_in_polygon(point, polygon):
    """
    Determines if a point is inside a polygon using the ray-casting algorithm.
    :param point: Tuple (x, y) representing the point.
    :param polygon: List of tuples [(x1, y1), (x2, y2), ...] representing polygon vertices.
    :return: True if the point is inside, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x if p1y != p2y else p1x
                if x <= x_intersect:
                    inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside
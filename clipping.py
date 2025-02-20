def clip_polygon(subject_polygon, clip_boundary):
    """Applies the Sutherland-Hodgman polygon clipping algorithm."""
    # If the subject polygon is empty, return an empty list immediately
    if not subject_polygon:
        return []
        
    def inside(p, edge):
        (x1, y1), (x2, y2) = edge
        return (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0
    
    def intersection(p1, p2, edge):
        (x1, y1), (x2, y2) = edge
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        
        if dx == 0 and dy == 0:
            return p1
        
        t = (((x1 - p1[0]) * (y2 - y1) - (y1 - p1[1]) * (x2 - x1)) /
             (dx * (y2 - y1) - dy * (x2 - x1)))
        
        return p1[0] + t * dx, p1[1] + t * dy
    
    output_list = subject_polygon
    # Create edges from the clipping boundary
    clip_edges = [(clip_boundary[i], clip_boundary[i+1]) for i in range(len(clip_boundary)-1)]
    
    for edge in clip_edges:
        input_list = output_list
        if not input_list:
            # The polygon has been fully clipped away
            return []
        output_list = []
        s = input_list[-1]
        
        for e in input_list:
            if inside(e, edge):
                if not inside(s, edge):
                    output_list.append(intersection(s, e, edge))
                output_list.append(e)
            elif inside(s, edge):
                output_list.append(intersection(s, e, edge))
            s = e
    
    return output_list
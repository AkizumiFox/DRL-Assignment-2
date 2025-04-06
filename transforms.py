# Board transformation functions for symmetry

def rot90(pattern):
    """Rotate pattern 90 degrees clockwise."""
    return [(y, 3 - x) for (x, y) in pattern]

def rot180(pattern):
    """Rotate pattern 180 degrees."""
    return [(3 - x, 3 - y) for (x, y) in pattern]

def rot270(pattern):
    """Rotate pattern 270 degrees clockwise."""
    return [(3 - y, x) for (x, y) in pattern]

def flip_horizontal(pattern):
    """Flip pattern horizontally."""
    return [(x, 3 - y) for (x, y) in pattern] 
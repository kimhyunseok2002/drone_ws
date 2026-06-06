"""Path loading / geometry helpers (pure-python, numpy optional).

The mission path is a CSV of ``x,y`` world-frame (ENU) waypoints with a fixed
0.1 m spacing, starting at (0, 0).  Altitude is NOT in the file; the drone holds
a constant cruise altitude (see params).  These helpers are deliberately
dependency-light so they can be unit tested without ROS.
"""
import csv
import math


def load_path_csv(path_file):
    """Load an ``x,y`` CSV into a list of (x, y) float tuples.

    Tolerates a header row, blank lines and extra columns.  Raises ValueError
    if no numeric rows are found.
    """
    pts = []
    with open(path_file, 'r') as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except (ValueError, IndexError):
                # header or malformed line -> skip
                continue
            pts.append((x, y))
    if len(pts) < 2:
        raise ValueError("path '%s' has fewer than 2 numeric points" % path_file)
    return pts


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap_to_pi(angle):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(a, b):
    """Smallest signed difference a-b wrapped to (-pi, pi]."""
    return wrap_to_pi(a - b)


def resample_polyline(pts, spacing):
    """Resample a polyline to (approximately) uniform ``spacing`` metres."""
    if not pts:
        return []
    out = [pts[0]]
    carry = 0.0
    for i in range(1, len(pts)):
        ax, ay = out[-1] if False else pts[i - 1]
        bx, by = pts[i]
        seg = math.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        ux, uy = (bx - ax) / seg, (by - ay) / seg
        d = spacing - carry
        while d <= seg:
            out.append((ax + ux * d, ay + uy * d))
            d += spacing
        carry = seg - (d - spacing)
    if dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out

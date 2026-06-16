"""Pure Pursuit path follower for a holonomic multirotor (2D, world/ENU frame).

Classic Pure Pursuit was formulated for car-like robots, but for a holonomic
drone we use the same geometric idea: pick a *lookahead point* on the path a
fixed arc-distance ahead of the vehicle and steer straight toward it.  The
controller here returns a desired travel *direction* (heading) in the world
frame; the caller turns that into a velocity command (and lets VFH veto the
direction when an obstacle is in the way).

Progress is tracked with a monotonically non-decreasing closest-point index so
the controller never snaps backward onto an earlier part of a self-approaching
path.
"""
import math

from .path_utils import dist, wrap_to_pi


class PurePursuit(object):
    def __init__(self, lookahead=1.0, goal_tolerance=0.3):
        self.lookahead = float(lookahead)
        self.goal_tolerance = float(goal_tolerance)
        self._path = []
        self._closest_idx = 0

    # ------------------------------------------------------------------ #
    def set_path(self, pts):
        """pts: iterable of (x, y).  Resets progress."""
        self._path = [(float(x), float(y)) for x, y in pts]
        self._closest_idx = 0

    @property
    def path(self):
        return self._path

    @property
    def goal(self):
        return self._path[-1] if self._path else None

    # ------------------------------------------------------------------ #
    def _advance_closest(self, pos):
        """Move the closest-point cursor forward (never backward)."""
        best_i = self._closest_idx
        best_d = dist(pos, self._path[best_i])
        # search a forward window so we keep up with the drone but never reverse
        i = self._closest_idx
        n = len(self._path)
        while i < n:
            d = dist(pos, self._path[i])
            if d < best_d:
                best_d = d
                best_i = i
            # stop scanning once we are clearly moving away again past lookahead
            if d > best_d + 2.0 * self.lookahead and i > best_i:
                break
            i += 1
        self._closest_idx = best_i
        return best_i, best_d

    def far_target(self, pos, lookahead):
        """Return a path point ~`lookahead` m ahead of the current closest point.

        Stable path-based goal (does not advance the cursor) — useful for planners
        like MPPI that need a goal roughly a horizon-distance ahead. Call after
        compute() so the closest-point cursor is up to date.
        """
        if not self._path:
            return pos
        n = len(self._path)
        j = self._closest_idx
        while j < n:
            if dist(pos, self._path[j]) >= lookahead:
                return self._path[j]
            j += 1
        return self._path[-1]

    def compute(self, pos):
        """Return a dict with the lookahead target and desired heading.

        Keys:
          target        : (x, y) lookahead point
          heading       : desired travel direction [rad] in world/ENU frame
          dist_to_goal  : straight-line distance to final path point
          progress      : closest_idx / (N-1)  in [0, 1]
          finished      : True when within goal_tolerance of the final point
        """
        if not self._path:
            raise RuntimeError("PurePursuit.compute called before set_path")

        ci, _ = self._advance_closest(pos)
        goal = self._path[-1]
        d_goal = dist(pos, goal)

        # Find the first point at least `lookahead` ahead of the closest point.
        target = goal
        n = len(self._path)
        j = ci
        while j < n:
            if dist(pos, self._path[j]) >= self.lookahead:
                target = self._path[j]
                break
            j += 1
        else:
            target = goal  # near the end: aim straight at the goal

        heading = math.atan2(target[1] - pos[1], target[0] - pos[0])
        finished = (ci >= n - 1) or (d_goal <= self.goal_tolerance)

        return {
            'target': target,
            'heading': wrap_to_pi(heading),
            'dist_to_goal': d_goal,
            'progress': float(ci) / float(max(1, n - 1)),
            'finished': finished,
        }

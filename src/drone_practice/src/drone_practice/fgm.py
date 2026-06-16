"""Follow-the-Gap Method (FGM) obstacle avoidance for a 2D LiDAR.

Idea (Sezer & Gokasan, 2012): instead of scoring directions like VFH, FGM
- finds the obstacles and inflates each by the robot radius + safety margin
  (a "safety bubble"), blocking a wedge around it,
- finds the largest remaining GAP (contiguous free angular sector), and
- steers toward the best point of a gap, blended with the goal direction.

Operates in the LiDAR/body frame (bearing 0 = drone nose), same interface as the
VFH module (returns a body-frame steering bearing), so it plugs into the same
position-carrot controller.  Pure-python (math only) -> unit testable.
"""
import math


class FGM(object):
    def __init__(self,
                 robot_radius=0.30,
                 safety_distance=0.35,
                 max_range=10.0,
                 active_distance=5.0,
                 min_gap_rad=0.35,
                 clearance_cone_rad=0.26):
        self.robot_radius = float(robot_radius)
        self.safety_distance = float(safety_distance)
        self.max_range = float(max_range)
        self.active_distance = float(active_distance)
        self.min_gap_rad = float(min_gap_rad)
        self.clearance_cone_rad = float(clearance_cone_rad)

    @staticmethod
    def _ang_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _clearance_along(self, proc, angle_min, inc, bearing):
        best = float('inf')
        for i, r in enumerate(proc):
            if abs(self._ang_diff(angle_min + i * inc, bearing)) <= self.clearance_cone_rad:
                if r < best:
                    best = r
        return best

    def compute(self, ranges, angle_min, angle_increment, target_dir):
        n = len(ranges)
        if n == 0:
            return {'steer': target_dir, 'free': True, 'clearance': float('inf')}

        # 1) clean ranges (no-return -> max_range)
        proc = []
        for r in ranges:
            if r is None or math.isinf(r) or math.isnan(r) or r <= 0.0:
                proc.append(self.max_range)
            else:
                proc.append(min(r, self.max_range))

        # 2) block a safety bubble (robot+safety) around every near obstacle
        enl = self.robot_radius + self.safety_distance
        blocked = [False] * n
        for i, r in enumerate(proc):
            if r > self.active_distance:
                continue
            half = math.pi / 2.0 if r <= enl else math.asin(max(-1.0, min(1.0, enl / r)))
            nb = int(math.ceil(half / angle_increment))
            for k in range(-nb, nb + 1):
                blocked[(i + k) % n] = True

        # no near obstacle -> straight to the goal
        if not any(blocked):
            return {'steer': target_dir, 'free': True,
                    'clearance': self._clearance_along(proc, angle_min, angle_increment, target_dir)}
        if all(blocked):
            return {'steer': None, 'free': False, 'clearance': 0.0}

        # 3) find gaps (contiguous free sectors), circular-safe (start at a blocked idx)
        start = next(i for i in range(n) if blocked[i])
        gaps, cur = [], []
        for k in range(n):
            i = (start + k) % n
            if not blocked[i]:
                cur.append(i)
            elif cur:
                gaps.append(cur); cur = []
        if cur:
            gaps.append(cur)

        min_gap_idx = max(1, int(math.ceil(self.min_gap_rad / angle_increment)))

        # 4) for each wide-enough gap pick a candidate bearing:
        #    if the goal points through the gap -> aim at the goal;
        #    else aim at the DEEPEST point of the gap (FGM hallmark).
        best_bearing, best_cost = None, float('inf')
        for g in gaps:
            if len(g) < min_gap_idx:
                continue
            bearings = [angle_min + i * angle_increment for i in g]
            lo, hi = bearings[0], bearings[-1]
            # goal inside this gap's span?
            if self._ang_diff(target_dir, lo) >= -1e-3 and self._ang_diff(hi, target_dir) >= -1e-3:
                cand = target_dir
            else:
                deepest = max(g, key=lambda i: proc[i])
                cand = angle_min + deepest * angle_increment
            cost = abs(self._ang_diff(cand, target_dir))
            if cost < best_cost:
                best_cost, best_bearing = cost, cand

        if best_bearing is None:
            return {'steer': None, 'free': False, 'clearance': 0.0}
        return {'steer': best_bearing, 'free': True,
                'clearance': self._clearance_along(proc, angle_min, angle_increment, best_bearing)}

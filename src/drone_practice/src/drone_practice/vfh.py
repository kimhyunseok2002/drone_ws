"""Vector Field Histogram (VFH) obstacle avoidance for a 2D LiDAR.

Operates entirely in the LiDAR / body frame:  bearing 0 = drone +x (nose),
angles increase counter-clockwise, matching ``sensor_msgs/LaserScan``.

Pipeline (Borenstein & Koren, 1991, simplified):
  1. Build a polar histogram over ``num_bins`` angular sectors.  Each LiDAR
     return within the active distance casts a magnitude vote into its sector
     and into neighbouring sectors, *enlarged* by the drone radius + safety
     margin so the planned path keeps real clearance.
  2. Threshold the histogram into free / blocked sectors.
  3. Among free sectors that belong to a gap wide enough for the drone, pick
     the bearing closest to the desired (Pure-Pursuit) target direction.

The result is a collision-free *body-frame* bearing the caller should steer
toward, plus the clearance along that bearing for speed scaling.
"""
import math


class VFH(object):
    def __init__(self,
                 num_bins=72,
                 threshold=2.0,
                 robot_radius=0.30,
                 safety_distance=0.35,
                 active_distance=8.0,
                 min_gap_rad=0.35,
                 clearance_cone_rad=0.26,
                 hysteresis_weight=0.35):
        self.num_bins = int(num_bins)
        self.bin_width = 2.0 * math.pi / self.num_bins
        self.threshold = float(threshold)
        self.robot_radius = float(robot_radius)
        self.safety_distance = float(safety_distance)
        self.active_distance = float(active_distance)
        self.min_gap_rad = float(min_gap_rad)
        self.clearance_cone_rad = float(clearance_cone_rad)
        # commitment: penalise switching away from the last chosen heading so the
        # drone picks a side of an obstacle and stays on it (avoids limit cycles)
        self.hysteresis_weight = float(hysteresis_weight)
        self.prev_steer = None
        self.last_histogram = [0.0] * self.num_bins

    # ------------------------------------------------------------------ #
    def _bin_of(self, angle):
        """Map an angle in (-pi, pi] to a bin index [0, num_bins)."""
        idx = int(math.floor((angle + math.pi) / self.bin_width))
        return idx % self.num_bins

    def _bin_center(self, idx):
        return -math.pi + (idx + 0.5) * self.bin_width

    @staticmethod
    def _ang_diff(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    # ------------------------------------------------------------------ #
    def build_histogram(self, ranges, angle_min, angle_increment):
        h = [0.0] * self.num_bins
        enlarge_r = self.robot_radius + self.safety_distance
        d_active = self.active_distance
        for i, r in enumerate(ranges):
            # reject no-return / out of active band
            if r is None:
                continue
            if math.isinf(r) or math.isnan(r):
                continue
            if r <= 0.0 or r > d_active:
                continue
            bearing = angle_min + i * angle_increment
            # proximity magnitude: closer -> larger vote
            mag = (d_active - r)
            # enlargement half-angle so the path clears the drone body
            if r <= enlarge_r:
                gamma = math.pi / 2.0
            else:
                gamma = math.asin(max(-1.0, min(1.0, enlarge_r / r)))
            # spread the vote over [bearing - gamma, bearing + gamma]
            n_spread = int(math.ceil(gamma / self.bin_width))
            center = self._bin_of(bearing)
            for k in range(-n_spread, n_spread + 1):
                h[(center + k) % self.num_bins] += mag
        self.last_histogram = h
        return h

    # ------------------------------------------------------------------ #
    def _clearance_along(self, ranges, angle_min, angle_increment, bearing):
        """Minimum LiDAR range within +/- clearance_cone of `bearing`."""
        best = float('inf')
        for i, r in enumerate(ranges):
            if r is None or math.isinf(r) or math.isnan(r) or r <= 0.0:
                continue
            b = angle_min + i * angle_increment
            if abs(self._ang_diff(b, bearing)) <= self.clearance_cone_rad:
                if r < best:
                    best = r
        return best

    # ------------------------------------------------------------------ #
    def compute(self, ranges, angle_min, angle_increment, target_dir):
        """Return the steering decision.

        target_dir : desired travel bearing in the body frame [rad].
        returns dict: steer (rad|None), free (bool), clearance (m), histogram.
        """
        h = self.build_histogram(ranges, angle_min, angle_increment)
        blocked = [v > self.threshold for v in h]

        # Trivial cases
        if not any(blocked):
            self.prev_steer = target_dir
            return {'steer': target_dir, 'free': True,
                    'clearance': self._clearance_along(ranges, angle_min,
                                                       angle_increment, target_dir),
                    'histogram': h}
        if all(blocked):
            return {'steer': None, 'free': False, 'clearance': 0.0, 'histogram': h}

        # Group contiguous free bins (start from a blocked bin to avoid wrap merge)
        n = self.num_bins
        start = next(i for i in range(n) if blocked[i])
        groups = []
        cur = []
        for k in range(n):
            i = (start + k) % n
            if not blocked[i]:
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)

        min_gap_bins = max(1, int(math.ceil(self.min_gap_rad / self.bin_width)))

        best_bin = None
        best_cost = float('inf')
        for g in groups:
            if len(g) < min_gap_bins:
                continue  # gap too narrow for the drone to fit
            for b in g:
                center = self._bin_center(b)
                cost = abs(self._ang_diff(center, target_dir))
                if self.prev_steer is not None:
                    # commitment term: prefer staying near the previously chosen
                    # heading so we don't flip sides of an obstacle each cycle
                    cost += self.hysteresis_weight * abs(
                        self._ang_diff(center, self.prev_steer))
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b

        if best_bin is None:
            # no gap wide enough -> hard blocked
            return {'steer': None, 'free': False, 'clearance': 0.0, 'histogram': h}

        steer = self._bin_center(best_bin)
        self.prev_steer = steer
        clr = self._clearance_along(ranges, angle_min, angle_increment, steer)
        return {'steer': steer, 'free': True, 'clearance': clr, 'histogram': h}

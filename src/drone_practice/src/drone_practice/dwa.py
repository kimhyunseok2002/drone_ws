"""Dynamic Window Approach (DWA) local planner for a holonomic multirotor.

Unlike VFH (which only picks a free *direction*), DWA reasons about the drone's
*dynamics*: it samples reachable velocity vectors inside a "dynamic window"
around the current velocity (bounded by acceleration), rolls each one forward
into a short predicted trajectory, rejects trajectories that would hit an
obstacle, and scores the survivors by

    score = w_head * heading-to-goal  +  w_clear * obstacle clearance
          + w_vel * forward speed

The winning velocity is dynamics-feasible, so the drone never commands a motion
its momentum can't safely follow -> no "drift into the obstacle", and the
trajectory-level scoring avoids the limit-cycle / wall-following that pure VFH
can fall into.

Everything is computed in the WORLD (ENU) frame so the output (vx, vy) can be
sent straight to a velocity setpoint.  Pure-python (math only) -> unit testable
without ROS.
"""
import math


class DWA(object):
    def __init__(self,
                 max_speed=1.0,
                 max_accel=2.0,
                 window_time=0.5,
                 predict_time=1.2,
                 dt_sim=0.1,
                 v_samples=7,
                 robot_radius=0.30,
                 safety_distance=0.35,
                 active_distance=5.0,
                 heading_weight=0.8,
                 clearance_weight=0.35,
                 velocity_weight=0.2,
                 ray_step=3):
        self.max_speed = float(max_speed)
        self.max_accel = float(max_accel)
        self.window_time = float(window_time)
        self.predict_time = float(predict_time)
        self.dt_sim = float(dt_sim)
        self.v_samples = int(v_samples)
        self.robot_radius = float(robot_radius)
        self.safety_distance = float(safety_distance)
        self.active_distance = float(active_distance)
        self.w_head = float(heading_weight)
        self.w_clear = float(clearance_weight)
        self.w_vel = float(velocity_weight)
        self.ray_step = max(1, int(ray_step))

    # ------------------------------------------------------------------ #
    def _obstacle_points(self, pos, yaw, ranges, angle_min, angle_inc):
        """LiDAR returns -> world-frame (x, y) obstacle points within range."""
        pts = []
        cy, sy = math.cos(yaw), math.sin(yaw)
        for i in range(0, len(ranges), self.ray_step):
            r = ranges[i]
            if r is None or math.isinf(r) or math.isnan(r):
                continue
            if r <= 0.0 or r > self.active_distance:
                continue
            b = angle_min + i * angle_inc          # body-frame bearing
            bx, by = r * math.cos(b), r * math.sin(b)
            # body -> world
            pts.append((pos[0] + cy * bx - sy * by,
                        pos[1] + sy * bx + cy * by))
        return pts

    # ------------------------------------------------------------------ #
    def _linspace(self, lo, hi, n):
        if n <= 1 or hi <= lo:
            return [0.5 * (lo + hi)]
        step = (hi - lo) / (n - 1)
        return [lo + step * k for k in range(n)]

    # ------------------------------------------------------------------ #
    def compute(self, pos, yaw, vel, ranges, angle_min, angle_inc, target_dir):
        """Return the best dynamics-feasible velocity.

        pos        : (x, y) world
        yaw        : heading [rad]
        vel        : (vx, vy) current world velocity
        target_dir : desired travel bearing [rad] (e.g. Pure-Pursuit heading)
        returns dict: vx, vy, ok (bool), clearance (m)
        """
        obs = self._obstacle_points(pos, yaw, ranges, angle_min, angle_inc)
        coll = self.robot_radius + self.safety_distance

        # dynamic window around current velocity, clamped to [-vmax, vmax]
        dv = self.max_accel * self.window_time
        vmax = self.max_speed
        vx_lo = max(-vmax, vel[0] - dv); vx_hi = min(vmax, vel[0] + dv)
        vy_lo = max(-vmax, vel[1] - dv); vy_hi = min(vmax, vel[1] + dv)
        vxs = self._linspace(vx_lo, vx_hi, self.v_samples)
        vys = self._linspace(vy_lo, vy_hi, self.v_samples)

        n_steps = max(1, int(self.predict_time / self.dt_sim))
        tdx, tdy = math.cos(target_dir), math.sin(target_dir)

        best = None
        best_score = -1e18
        best_clear = 0.0
        # always consider a full stop (braking) as a safe fallback candidate
        candidates = [(vx, vy) for vx in vxs for vy in vys]
        candidates.append((0.0, 0.0))

        for (vx, vy) in candidates:
            speed = math.hypot(vx, vy)
            if speed > vmax + 1e-6:
                continue

            # roll out trajectory, find closest obstacle approach
            min_clear = self.active_distance
            hit = False
            for k in range(1, n_steps + 1):
                tx = pos[0] + vx * k * self.dt_sim
                ty = pos[1] + vy * k * self.dt_sim
                for (ox, oy) in obs:
                    d = math.hypot(tx - ox, ty - oy)
                    if d < min_clear:
                        min_clear = d
                    if d <= coll:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                continue

            # --- scores (normalised to ~[0,1]) ---
            if speed < 1e-3:
                heading = 0.0          # discourage stopping unless forced
            else:
                cosang = (vx * tdx + vy * tdy) / speed
                heading = 0.5 * (cosang + 1.0)
            clearance = min(min_clear, self.active_distance) / self.active_distance
            velocity = speed / vmax

            score = (self.w_head * heading +
                     self.w_clear * clearance +
                     self.w_vel * velocity)
            if score > best_score:
                best_score = score
                best = (vx, vy)
                best_clear = min_clear

        if best is None:
            return {'vx': 0.0, 'vy': 0.0, 'ok': False, 'clearance': 0.0}
        return {'vx': best[0], 'vy': best[1], 'ok': True, 'clearance': best_clear}

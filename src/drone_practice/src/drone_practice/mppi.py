"""Model Predictive Path Integral (MPPI) local planner for a holonomic drone.

MPPI is a sampling-based MPC: every cycle it draws K random control sequences
around a warm-started nominal control, rolls each one forward over a horizon,
scores the trajectories with a cost (obstacle proximity + distance to the goal),
and updates the control as a softmax(-cost/lambda)-weighted average of the
samples.  The first control of the updated sequence is executed.

It optimises path-following and obstacle-avoidance jointly in one objective, so
there is no separate "follower vs avoider" layer to fight each other.

World/ENU frame; control = velocity (vx, vy) -> sent straight to a velocity
setpoint.  Uses numpy for vectorised rollouts (real-time at 20 Hz).
"""
import math
import numpy as np


class MPPI(object):
    def __init__(self,
                 horizon=20,
                 samples=300,
                 dt=0.1,
                 lambda_=1.0,
                 noise_sigma=0.40,
                 max_speed=1.0,
                 robot_radius=0.30,
                 safety_distance=0.35,
                 active_distance=5.0,
                 repel_distance=1.5,
                 ray_step=4,
                 w_obstacle=40.0,
                 w_goal=1.0,
                 collision_cost=1.0e4):
        self.N = int(horizon)
        self.K = int(samples)
        self.dt = float(dt)
        self.lam = float(lambda_)
        self.sigma = float(noise_sigma)
        self.vmax = float(max_speed)
        self.coll = float(robot_radius + safety_distance)
        self.active = float(active_distance)
        self.repel = float(repel_distance)
        self.ray_step = max(1, int(ray_step))
        self.w_obs = float(w_obstacle)
        self.w_goal = float(w_goal)
        self.c_coll = float(collision_cost)
        self.U = np.zeros((self.N, 2))      # warm-started nominal control

    # ------------------------------------------------------------------ #
    def _obstacles(self, pos, yaw, ranges, angle_min, inc):
        cy, sy = math.cos(yaw), math.sin(yaw)
        pts = []
        for i in range(0, len(ranges), self.ray_step):
            r = ranges[i]
            if r is None or math.isinf(r) or math.isnan(r) or r <= 0.0 or r > self.active:
                continue
            b = angle_min + i * inc
            bx, by = r * math.cos(b), r * math.sin(b)
            pts.append((pos[0] + cy * bx - sy * by, pos[1] + sy * bx + cy * by))
        return np.array(pts) if pts else np.zeros((0, 2))

    # ------------------------------------------------------------------ #
    def compute(self, pos, yaw, ranges, angle_min, angle_increment, target_point):
        pos = np.array(pos, dtype=float)
        target = np.array(target_point, dtype=float)
        O = self._obstacles(pos, yaw, ranges, angle_min, angle_increment)

        # sample controls around the nominal: V = U + noise, clipped to vmax box
        noise = np.random.normal(0.0, self.sigma, size=(self.K, self.N, 2))
        V = self.U[None, :, :] + noise
        np.clip(V, -self.vmax, self.vmax, out=V)

        # rollout (kinematic velocity model): P[k,t] = pos + dt * cumsum(V)
        P = pos[None, None, :] + np.cumsum(V, axis=1) * self.dt   # (K, N, 2)

        # --- cost ---
        # goal: distance of every rollout point to the look-ahead target
        gdist = np.linalg.norm(P - target[None, None, :], axis=2)  # (K, N)
        cost = self.w_goal * gdist.sum(axis=1)                      # (K,)

        # obstacle: penalise proximity; huge cost if a point enters collision
        if O.shape[0] > 0:
            d = np.linalg.norm(P[:, :, None, :] - O[None, None, :, :], axis=3)  # (K,N,M)
            dmin = d.min(axis=2)                                    # (K, N)
            # only repel when CLOSE (within repel distance); far obstacles must
            # not overwhelm the goal term, otherwise the drone flees prematurely.
            prox = np.clip(self.repel - dmin, 0.0, self.repel)
            obs_cost = prox.sum(axis=1) + self.c_coll * (dmin < self.coll).sum(axis=1)
            cost = cost + self.w_obs * obs_cost

        # softmax(-cost/lambda) weighting
        beta = cost.min()
        w = np.exp(-(cost - beta) / self.lam)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum < 1e-9:
            return {'vx': 0.0, 'vy': 0.0, 'ok': False}
        w /= w_sum

        U_new = np.einsum('k,knd->nd', w, V)        # weighted-average control seq
        u0 = U_new[0].copy()
        # warm start: shift the sequence forward
        self.U[:-1] = U_new[1:]
        self.U[-1] = U_new[-1]

        # clamp executed command to speed limit
        sp = float(np.hypot(u0[0], u0[1]))
        if sp > self.vmax and sp > 1e-9:
            u0 *= self.vmax / sp
        return {'vx': float(u0[0]), 'vy': float(u0[1]), 'ok': True}

    def reset(self):
        self.U[:] = 0.0

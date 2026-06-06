#!/usr/bin/env python3
"""Autonomous mission node: takeoff -> Pure Pursuit + VFH -> precision landing.

Phases (single OFFBOARD session, MAVROS):
  TAKEOFF : climb to cruise altitude over the start point (0,0).
  FOLLOW  : Pure Pursuit picks a lookahead point on the CSV path and yields a
            desired heading; VFH (single 2D LiDAR) vetoes that heading whenever
            an obstacle is in the way, returning a collision-free body bearing.
            Horizontal motion is velocity-controlled; altitude is held with a P
            controller (rule: never climb over obstacles).
  LAND    : centre over the final path point (the landing pad) and AUTO.LAND.

Frames: MAVROS local pose & setpoints are ENU. LaserScan is in the body frame
(bearing 0 = drone nose). The two are bridged with the live yaw from the pose.
"""
import math

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from mavros_msgs.msg import State, PositionTarget, ParamValue
from mavros_msgs.srv import CommandBool, SetMode, ParamSet

from drone_practice.pure_pursuit import PurePursuit
from drone_practice.vfh import VFH
from drone_practice.dwa import DWA
from drone_practice.path_utils import load_path_csv, wrap_to_pi


# PositionTarget.type_mask bit groups -------------------------------------------------
PT = PositionTarget
MASK_USE_POS_YAW = (PT.IGNORE_VX | PT.IGNORE_VY | PT.IGNORE_VZ |
                    PT.IGNORE_AFX | PT.IGNORE_AFY | PT.IGNORE_AFZ | PT.IGNORE_YAW_RATE)
MASK_USE_VEL_YAW = (PT.IGNORE_PX | PT.IGNORE_PY | PT.IGNORE_PZ |
                    PT.IGNORE_AFX | PT.IGNORE_AFY | PT.IGNORE_AFZ | PT.IGNORE_YAW_RATE)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class MissionNode(object):
    def __init__(self):
        rospy.init_node('mission_node')

        # ---- params -------------------------------------------------------
        g = lambda n, d: rospy.get_param('~' + n, d)
        self.cruise_alt = g('cruise_altitude', 2.5)
        self.takeoff_alt = g('takeoff_altitude', 2.5)
        self.arena = g('arena_limit', 25.0)
        self.goal_tol = g('goal_tolerance', 0.35)
        self.takeoff_tol = g('takeoff_reached_tol', 0.15)

        self.cruise_speed = g('cruise_speed', 1.2)
        self.min_speed = g('min_speed', 0.35)
        self.slow_clr = g('slow_down_clearance', 2.5)
        self.estop_clr = g('emergency_stop_clearance', 0.8)

        self.kp_alt = g('kp_altitude', 1.2)
        self.max_vz = g('max_vz', 0.8)
        self.yaw_mode = g('yaw_mode', 'fixed')
        self.yaw_slew = g('yaw_slew_rate', 1.0)

        self.land_speed = g('land_descend_speed', 0.4)
        self.land_settle = g('land_settle_radius', 0.20)
        self.land_auto = g('land_use_auto_land', True)

        self.rate_hz = g('control_rate', 20.0)
        scan_topic = g('scan_topic', '/scan')
        self.disable_rc_failsafe = g('disable_rc_failsafe', True)

        path_csv = rospy.get_param('~path_csv')   # required
        pts = load_path_csv(path_csv)
        rospy.loginfo('mission: loaded %d path points from %s', len(pts), path_csv)

        self.pp = PurePursuit(lookahead=g('lookahead_distance', 0.9),
                              goal_tolerance=self.goal_tol)
        self.pp.set_path(pts)
        self.goal = pts[-1]

        self.vfh = VFH(num_bins=int(g('vfh_num_bins', 72)),
                       threshold=g('vfh_threshold', 2.0),
                       robot_radius=g('vfh_robot_radius', 0.30),
                       safety_distance=g('vfh_safety_distance', 0.35),
                       active_distance=g('vfh_active_distance', 7.0),
                       min_gap_rad=g('vfh_min_gap_rad', 0.35),
                       clearance_cone_rad=g('vfh_clearance_cone_rad', 0.26),
                       hysteresis_weight=g('vfh_hysteresis_weight', 0.35))

        # obstacle-avoidance backend: 'vfh' (default) or 'dwa'
        self.avoidance = g('obstacle_avoidance', 'vfh').lower()
        self.dwa = DWA(max_speed=g('dwa_max_speed', 1.0),
                       max_accel=g('dwa_max_accel', 2.0),
                       window_time=g('dwa_window_time', 0.5),
                       predict_time=g('dwa_predict_time', 1.2),
                       dt_sim=g('dwa_dt_sim', 0.1),
                       v_samples=int(g('dwa_v_samples', 7)),
                       robot_radius=g('vfh_robot_radius', 0.30),
                       safety_distance=g('vfh_safety_distance', 0.35),
                       active_distance=g('dwa_active_distance', 5.0),
                       heading_weight=g('dwa_heading_weight', 0.8),
                       clearance_weight=g('dwa_clearance_weight', 0.35),
                       velocity_weight=g('dwa_velocity_weight', 0.2))
        self.max_steer_rate = g('max_steer_rate', 2.5)   # [rad/s] heading slew limit
        self.steer_cmd = None

        # control mode: 'position' uses a carrot position setpoint (drift-free,
        # PX4 position loop holds firmly); 'velocity' is the legacy mode.
        self.control_mode = g('control_mode', 'position')
        self.carrot_distance = g('carrot_distance', 1.2)  # [m] position lookahead
        self.min_carrot = g('min_carrot', 0.25)           # [m] keep creeping in tight gaps
        self.mpc_xy_cruise = g('mpc_xy_cruise', 1.2)      # [m/s] PX4 cruise cap
        self.mpc_xy_vel_max = g('mpc_xy_vel_max', 2.0)    # [m/s] PX4 max horiz speed

        # ---- state --------------------------------------------------------
        self.state = State()
        self.pose = None
        self.scan = None
        self.vel_world = (0.0, 0.0)   # current horizontal velocity (ENU), for DWA
        self.yaw_cmd = 0.0
        self.phase = 'TAKEOFF'

        # ---- io -----------------------------------------------------------
        rospy.Subscriber('/mavros/state', State, self._state_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self._pose_cb, queue_size=1)
        rospy.Subscriber(scan_topic, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped,
                         self._vel_cb, queue_size=1)

        self.sp_pub = rospy.Publisher('/mavros/setpoint_raw/local',
                                      PositionTarget, queue_size=10)
        self.path_pub = rospy.Publisher('~path', Path, queue_size=1, latch=True)
        self._publish_path_viz(pts)

        rospy.loginfo('mission: waiting for MAVROS services...')
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/param/set')
        self.arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.param_srv = rospy.ServiceProxy('/mavros/param/set', ParamSet)

        self._last_req = rospy.Time(0)

    # ------------------------------------------------------------------ #
    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def _scan_cb(self, msg):
        self.scan = msg

    def _vel_cb(self, msg):
        self.vel_world = (msg.twist.linear.x, msg.twist.linear.y)

    def _publish_path_viz(self, pts):
        m = Path()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        for (x, y) in pts:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = self.cruise_alt
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        self.path_pub.publish(m)

    # ------------------------------------------------------------------ #
    def _make_pos_sp(self, x, y, z, yaw):
        sp = PositionTarget()
        sp.header.stamp = rospy.Time.now()
        sp.coordinate_frame = PT.FRAME_LOCAL_NED   # MAVROS expects ENU here
        sp.type_mask = MASK_USE_POS_YAW
        sp.position.x = x
        sp.position.y = y
        sp.position.z = z
        sp.yaw = yaw
        return sp

    def _make_vel_sp(self, vx, vy, vz, yaw):
        sp = PositionTarget()
        sp.header.stamp = rospy.Time.now()
        sp.coordinate_frame = PT.FRAME_LOCAL_NED
        sp.type_mask = MASK_USE_VEL_YAW
        sp.velocity.x = vx
        sp.velocity.y = vy
        sp.velocity.z = vz
        sp.yaw = yaw
        return sp

    # ------------------------------------------------------------------ #
    def _ensure_offboard_armed(self):
        """Retry OFFBOARD + arm at most every 2 s (standard PX4 pattern)."""
        now = rospy.Time.now()
        if (now - self._last_req).to_sec() < 2.0:
            return
        self._last_req = now
        if self.state.mode != 'OFFBOARD':
            try:
                self.mode_srv(base_mode=0, custom_mode='OFFBOARD')
                rospy.loginfo_throttle(5, 'mission: requesting OFFBOARD')
            except rospy.ServiceException as e:
                rospy.logwarn('set_mode failed: %s', e)
        elif not self.state.armed:
            try:
                self.arm_srv(True)
                rospy.loginfo_throttle(5, 'mission: arming')
            except rospy.ServiceException as e:
                rospy.logwarn('arming failed: %s', e)

    def _set_px4_param(self, name, value):
        """Set a PX4 param via MAVROS (int or float), retrying until the param
        plugin has synced. Returns True on success."""
        is_int = isinstance(value, int)
        pv = ParamValue(integer=int(value) if is_int else 0,
                        real=0.0 if is_int else float(value))
        for _ in range(30):                       # ~15 s patience for param sync
            try:
                if self.param_srv(param_id=name, value=pv).success:
                    return True
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(2, 'param set %s failed: %s', name, e)
            rospy.sleep(0.5)
        return False

    def _configure_px4(self):
        """Prepare PX4 for an autonomous OFFBOARD mission:
          * disable the SITL RC/datalink-loss failsafe (no RC -> would force RTL
            and block OFFBOARD; config change, NOT manual piloting), and
          * cap horizontal speed so position-carrot motion stays gentle and
            drift-free (the main fix for 'floaty' overshoot into obstacles).
        """
        params = []
        if self.disable_rc_failsafe:
            # NAV_RCL_ACT=0 (disable RC-loss action) is critical, set first.
            params += [('NAV_RCL_ACT', 0), ('NAV_DLL_ACT', 0), ('COM_RCL_EXCEPT', 4)]
        # cap horizontal speed for both carrot-position and DWA velocity control
        params += [('MPC_XY_CRUISE', float(self.mpc_xy_cruise)),
                   ('MPC_XY_VEL_MAX', float(self.mpc_xy_vel_max))]
        for name, val in params:
            ok = self._set_px4_param(name, val)
            rospy.loginfo('mission: %s=%s -> %s', name, val, 'OK' if ok else 'FAILED')

    def _set_mode(self, mode):
        try:
            return self.mode_srv(base_mode=0, custom_mode=mode).mode_sent
        except rospy.ServiceException as e:
            rospy.logwarn('set_mode(%s) failed: %s', mode, e)
            return False

    # ------------------------------------------------------------------ #
    def _yaw_target(self, travel_dir, dt):
        if self.yaw_mode == 'travel':
            err = wrap_to_pi(travel_dir - self.yaw_cmd)
            step = clamp(err, -self.yaw_slew * dt, self.yaw_slew * dt)
            self.yaw_cmd = wrap_to_pi(self.yaw_cmd + step)
        else:
            self.yaw_cmd = 0.0
        return self.yaw_cmd

    def _follow_setpoint(self, dt):
        """Dispatch to the selected obstacle-avoidance backend."""
        if self.avoidance == 'dwa':
            return self._follow_dwa(dt)
        return self._follow_vfh(dt)

    def _follow_dwa(self, dt):
        """Pure Pursuit (goal heading) + DWA (dynamics-aware avoidance) -> a
        velocity setpoint. DWA already respects acceleration limits, so the
        commanded velocity is feasible and the drone does not drift into
        obstacles."""
        p = self.pose.pose.position
        yaw = yaw_from_quat(self.pose.pose.orientation)
        pos = (p.x, p.y)
        pp = self.pp.compute(pos)
        target_dir = pp['heading']

        if self.scan is None:
            # no perception yet: head straight to the lookahead at cruise speed
            speed = min(self.dwa.max_speed, max(0.0, 1.5 * pp['dist_to_goal']))
            vx, vy = speed * math.cos(target_dir), speed * math.sin(target_dir)
        else:
            res = self.dwa.compute(pos, yaw, self.vel_world,
                                   list(self.scan.ranges),
                                   self.scan.angle_min, self.scan.angle_increment,
                                   target_dir)
            vx, vy = res['vx'], res['vy']
            # ease off near the goal so we don't overshoot the pad
            cap = max(0.0, 1.5 * pp['dist_to_goal'])
            sp = math.hypot(vx, vy)
            if sp > cap and sp > 1e-6:
                vx, vy = vx * cap / sp, vy * cap / sp

        vz = clamp(self.kp_alt * (self.cruise_alt - p.z), -self.max_vz, self.max_vz)
        travel = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-3 else target_dir
        yaw_cmd = self._yaw_target(travel, dt)
        return self._make_vel_sp(vx, vy, vz, yaw_cmd), pp['finished']

    def _follow_vfh(self, dt):
        """Pure Pursuit + VFH -> setpoint. Returns (sp, finished).

        Default 'position' mode places a *carrot* position target a short
        distance ahead along the collision-free heading; PX4's position loop
        holds firmly (no velocity drift) and decelerates near obstacles because
        the carrot shrinks with clearance. 'velocity' mode is the legacy path.
        """
        p = self.pose.pose.position
        yaw = yaw_from_quat(self.pose.pose.orientation)
        pos = (p.x, p.y)

        pp = self.pp.compute(pos)
        desired_world = pp['heading']

        # --- VFH veto in body frame ---
        steer_world = desired_world
        clearance = float('inf')
        blocked = False
        if self.scan is not None:
            target_body = wrap_to_pi(desired_world - yaw)
            res = self.vfh.compute(list(self.scan.ranges),
                                   self.scan.angle_min,
                                   self.scan.angle_increment,
                                   target_body)
            clearance = res['clearance']
            if res['steer'] is None:
                blocked = True
            else:
                steer_world = wrap_to_pi(res['steer'] + yaw)

        # --- heading slew limit: don't reverse direction instantly ---
        if self.steer_cmd is None:
            self.steer_cmd = steer_world
        else:
            derr = wrap_to_pi(steer_world - self.steer_cmd)
            derr = clamp(derr, -self.max_steer_rate * dt, self.max_steer_rate * dt)
            self.steer_cmd = wrap_to_pi(self.steer_cmd + derr)
        steer_world = self.steer_cmd
        yaw_cmd = self._yaw_target(steer_world, dt)

        # ============================ position (carrot) =================== #
        if self.control_mode == 'position':
            margin = self.vfh.robot_radius + self.vfh.safety_distance
            if blocked or clearance < self.estop_clr:
                carrot = 0.0                       # hold position firmly
            else:
                carrot = clamp(clearance - margin, 0.0, self.carrot_distance)
                carrot = max(carrot, self.min_carrot)      # keep creeping
                carrot = min(carrot, pp['dist_to_goal'])   # don't overshoot goal
            tx = p.x + carrot * math.cos(steer_world)
            ty = p.y + carrot * math.sin(steer_world)
            return self._make_pos_sp(tx, ty, self.cruise_alt, yaw_cmd), pp['finished']

        # ============================ velocity (legacy) ================== #
        if blocked:
            vz = clamp(self.kp_alt * (self.cruise_alt - p.z), -self.max_vz, self.max_vz)
            return self._make_vel_sp(0.0, 0.0, vz, yaw_cmd), False
        if clearance < self.estop_clr:
            speed = 0.0
        elif clearance < self.slow_clr:
            frac = (clearance - self.estop_clr) / max(1e-3, self.slow_clr - self.estop_clr)
            speed = max(self.min_speed, self.cruise_speed * frac)
        else:
            speed = self.cruise_speed
        speed = min(speed, max(self.min_speed, 1.5 * pp['dist_to_goal']))
        vx = speed * math.cos(steer_world)
        vy = speed * math.sin(steer_world)
        vz = clamp(self.kp_alt * (self.cruise_alt - p.z), -self.max_vz, self.max_vz)
        return self._make_vel_sp(vx, vy, vz, yaw_cmd), pp['finished']

    # ------------------------------------------------------------------ #
    def _arena_ok(self):
        if self.pose is None:
            return True
        p = self.pose.pose.position
        return abs(p.x) <= self.arena and abs(p.y) <= self.arena

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz

        # Wait for FCU connection + first pose.
        while not rospy.is_shutdown() and (not self.state.connected or self.pose is None):
            self.sp_pub.publish(self._make_pos_sp(0, 0, self.takeoff_alt, 0.0))
            rate.sleep()
        rospy.loginfo('mission: FCU connected, pose received. Starting.')

        # Make OFFBOARD survivable in SITL (no RC) + cap speed before arming.
        self._configure_px4()

        land_requested = False
        while not rospy.is_shutdown():
            p = self.pose.pose.position

            if not self._arena_ok():
                rospy.logwarn_throttle(2, 'mission: outside arena -> landing')
                self.phase = 'LAND'

            if self.phase == 'TAKEOFF':
                self._ensure_offboard_armed()
                self.sp_pub.publish(self._make_pos_sp(0, 0, self.takeoff_alt, 0.0))
                if (self.state.armed and self.state.mode == 'OFFBOARD' and
                        abs(p.z - self.takeoff_alt) <= self.takeoff_tol):
                    rospy.loginfo('mission: takeoff complete (z=%.2f). FOLLOW.', p.z)
                    self.phase = 'FOLLOW'

            elif self.phase == 'FOLLOW':
                sp, finished = self._follow_setpoint(dt)
                self.sp_pub.publish(sp)
                if finished:
                    rospy.loginfo('mission: reached final path point. LAND.')
                    self.phase = 'LAND'

            elif self.phase == 'LAND':
                gx, gy = self.goal
                dxy = math.hypot(p.x - gx, p.y - gy)
                if not (self.land_auto and land_requested):
                    # hold over the pad at cruise altitude until centred
                    self.sp_pub.publish(
                        self._make_pos_sp(gx, gy, self.cruise_alt, self.yaw_cmd))
                    if dxy <= self.land_settle:
                        if self.land_auto:
                            if self._set_mode('AUTO.LAND'):
                                rospy.loginfo('mission: centred over pad -> AUTO.LAND')
                                land_requested = True
                        else:
                            self.phase = 'MANUAL_LAND'
                else:
                    # AUTO.LAND engaged: keep nudging a setpoint in case we fall
                    # back to OFFBOARD, and watch for disarm / ground contact.
                    self.sp_pub.publish(
                        self._make_pos_sp(gx, gy, max(0.0, p.z - 0.5), self.yaw_cmd))
                    if (not self.state.armed) or p.z < 0.20:
                        rospy.loginfo('mission: landed and disarmed. DONE.')
                        break

            elif self.phase == 'MANUAL_LAND':
                gx, gy = self.goal
                z_cmd = max(0.0, p.z - self.land_speed * dt)
                self.sp_pub.publish(self._make_pos_sp(gx, gy, z_cmd, self.yaw_cmd))
                if p.z < 0.25:
                    try:
                        self.arm_srv(False)
                    except rospy.ServiceException:
                        pass
                    rospy.loginfo('mission: manual landing complete. DONE.')
                    break

            rate.sleep()


if __name__ == '__main__':
    try:
        MissionNode().run()
    except rospy.ROSInterruptException:
        pass

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
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Path
from mavros_msgs.msg import State, PositionTarget, ParamValue
from mavros_msgs.srv import CommandBool, SetMode, ParamSet

# OpenCV + cv_bridge are only needed for vision-guided landing. Import lazily so
# the node still runs (falling back to position-based landing) if they're absent.
try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    _HAVE_CV = True
except Exception as _cv_err:        # pragma: no cover
    _HAVE_CV = False
    _CV_IMPORT_ERR = _cv_err

from drone_practice.pure_pursuit import PurePursuit
from drone_practice.vfh import VFH
from drone_practice.dwa import DWA
from drone_practice.fgm import FGM
from drone_practice.mppi import MPPI
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

        # ---- vision-guided precision landing (down camera + red-pad servo) ----
        self.vision_landing = bool(g('vision_landing', True)) and _HAVE_CV
        if g('vision_landing', True) and not _HAVE_CV:
            rospy.logwarn('mission: vision_landing requested but OpenCV/cv_bridge '
                          'unavailable (%s) -> position-based landing fallback',
                          _CV_IMPORT_ERR)
        self.cam_topic = g('camera_topic', '/landing_cam/image_raw')
        self.v_kp = g('vision_kp', 0.9)               # [m/s per norm-err] servo gain
        self.v_max = g('vision_max_speed', 0.6)       # [m/s] horiz servo cap
        self.v_align_tol = g('vision_align_tol', 0.08)  # [norm] centred-enough to descend
        self.v_descend = g('vision_descend_speed', 0.35)  # [m/s] descent when centred
        self.v_handoff_alt = g('vision_handoff_alt', 0.35)  # [m] hand to AUTO.LAND
        self.v_lost_timeout = g('vision_lost_timeout', 1.0)  # [s] stale-detection cutoff
        self.v_min_area = g('vision_min_area', 80.0)  # [px^2] reject tiny red noise
        self.v_fallback_after = g('vision_fallback_after', 8.0)  # [s] no-pad -> pos land
        self.v_sign_x = g('vision_sign_x', -1.0)      # image->world sign (flip if diverges)
        self.v_sign_y = g('vision_sign_y', -1.0)
        self.vision_debug = bool(g('vision_debug', True))  # publish annotated image
        self.debug_topic = g('vision_debug_topic', '/landing_cam/debug')

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

        # obstacle-avoidance backend: 'vfh' (default) | 'dwa' | 'fgm' | 'mppi'
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
        self.fgm = FGM(robot_radius=g('vfh_robot_radius', 0.30),
                       safety_distance=g('vfh_safety_distance', 0.35),
                       active_distance=g('fgm_active_distance', 5.0),
                       min_gap_rad=g('fgm_min_gap_rad', 0.35),
                       clearance_cone_rad=g('vfh_clearance_cone_rad', 0.26))
        self.mppi = MPPI(horizon=int(g('mppi_horizon', 20)),
                         samples=int(g('mppi_samples', 300)),
                         dt=g('mppi_dt', 0.1),
                         lambda_=g('mppi_lambda', 1.0),
                         noise_sigma=g('mppi_noise_sigma', 0.40),
                         max_speed=g('mppi_max_speed', 1.0),
                         robot_radius=g('vfh_robot_radius', 0.30),
                         safety_distance=g('vfh_safety_distance', 0.35),
                         active_distance=g('mppi_active_distance', 5.0),
                         repel_distance=g('mppi_repel_distance', 1.5),
                         w_obstacle=g('mppi_w_obstacle', 40.0),
                         w_goal=g('mppi_w_goal', 1.0))
        self.mppi_goal_lookahead = g('mppi_goal_lookahead', 2.5)  # [m] goal projection
        # direction-based planners (vfh/fgm) share the carrot controller
        self.dir_planner = self.fgm if self.avoidance == 'fgm' else self.vfh
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

        # vision-landing detection state (updated by the camera callback)
        self.bridge = CvBridge() if self.vision_landing else None
        self.pad_err_u = 0.0          # [-1..1] horizontal pixel error (right = +)
        self.pad_err_v = 0.0          # [-1..1] vertical pixel error (down  = +)
        self.pad_err_norm = 1.0       # hypot of the two
        self.pad_seen_time = rospy.Time(0)
        self.land_t0 = None           # set when the LAND phase first begins
        self.debug_pub = None         # set in io block if vision_debug is on

        # ---- io -----------------------------------------------------------
        rospy.Subscriber('/mavros/state', State, self._state_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped,
                         self._pose_cb, queue_size=1)
        rospy.Subscriber(scan_topic, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped,
                         self._vel_cb, queue_size=1)
        if self.vision_landing:
            if self.vision_debug:
                self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)
            rospy.Subscriber(self.cam_topic, Image, self._image_cb, queue_size=1,
                             buff_size=2 ** 22)
            rospy.loginfo('mission: vision landing ON, camera=%s, debug=%s',
                          self.cam_topic, self.debug_topic if self.vision_debug else 'off')

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

    # ------------------------------------------------------------------ #
    def _image_cb(self, msg):
        """Detect the red landing-pad target in the down-camera image and store
        the normalised pixel error of its centroid from the image centre.
        u: +1 = right edge, v: +1 = bottom edge. When vision_debug is on, an
        annotated copy is republished on the debug topic for rqt_image_view."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logwarn_throttle(5, 'mission: cv_bridge failed: %s', e)
            return
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # red wraps the hue circle -> two bands
        m1 = cv2.inRange(hsv, (0, 90, 70), (10, 255, 255))
        m2 = cv2.inRange(hsv, (160, 90, 70), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found, cx, cy, c = False, w / 2.0, h / 2.0, None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if cv2.contourArea(c) >= self.v_min_area and M['m00'] > 0.0:
                cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                self.pad_err_u = (cx - w / 2.0) / (w / 2.0)
                self.pad_err_v = (cy - h / 2.0) / (h / 2.0)
                self.pad_err_norm = math.hypot(self.pad_err_u, self.pad_err_v)
                self.pad_seen_time = rospy.Time.now()
                found = True

        if self.debug_pub is not None:
            self._publish_debug(img, found, cx, cy, c)

    def _publish_debug(self, img, found, cx, cy, contour):
        """Draw the image-centre crosshair, the detected pad centroid and the
        error vector, then republish for visualisation."""
        h, w = img.shape[:2]
        ctr = (int(w / 2), int(h / 2))
        # image centre crosshair (cyan) + centred tolerance box (green)
        cv2.line(img, (ctr[0] - 20, ctr[1]), (ctr[0] + 20, ctr[1]), (255, 255, 0), 1)
        cv2.line(img, (ctr[0], ctr[1] - 20), (ctr[0], ctr[1] + 20), (255, 255, 0), 1)
        tol = int(self.v_align_tol * (w / 2.0))
        cv2.rectangle(img, (ctr[0] - tol, ctr[1] - tol),
                      (ctr[0] + tol, ctr[1] + tol), (0, 200, 0), 1)
        if found:
            cen = (int(cx), int(cy))
            if contour is not None:
                cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            cv2.circle(img, cen, 6, (0, 0, 255), -1)          # centroid (red)
            cv2.arrowedLine(img, ctr, cen, (0, 165, 255), 2)  # error vector (orange)
            ok = self.pad_err_norm < self.v_align_tol
            txt = 'CENTERED' if ok else 'aligning'
            col = (0, 220, 0) if ok else (0, 165, 255)
            cv2.putText(img, '%s  err=%.3f (u=%.2f v=%.2f)' %
                        (txt, self.pad_err_norm, self.pad_err_u, self.pad_err_v),
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        else:
            cv2.putText(img, 'NO PAD', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        except Exception as e:
            rospy.logwarn_throttle(5, 'mission: debug publish failed: %s', e)

    def _pad_fresh(self):
        """True if the pad was detected within the lost-detection timeout."""
        return (rospy.Time.now() - self.pad_seen_time).to_sec() < self.v_lost_timeout

    def _vision_servo(self, yaw):
        """Map the stored pixel error to an ENU horizontal velocity that drives
        the drone over the pad. The down camera (pitched +90 deg about Y) maps
        image-down (v) -> body -X and image-right (u) -> body -Y; the body
        velocity is then rotated to world by the live yaw (held ~0)."""
        vbx = self.v_sign_x * self.v_kp * self.pad_err_v
        vby = self.v_sign_y * self.v_kp * self.pad_err_u
        vx = vbx * math.cos(yaw) - vby * math.sin(yaw)
        vy = vbx * math.sin(yaw) + vby * math.cos(yaw)
        sp = math.hypot(vx, vy)
        if sp > self.v_max and sp > 1e-6:
            vx, vy = vx * self.v_max / sp, vy * self.v_max / sp
        return vx, vy

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
        if self.avoidance == 'mppi':
            return self._follow_mppi(dt)
        return self._follow_vfh(dt)        # vfh or fgm (direction-based)

    def _follow_mppi(self, dt):
        """Pure Pursuit (look-ahead goal) + MPPI -> velocity setpoint. MPPI
        optimises following + avoidance jointly over sampled trajectories."""
        p = self.pose.pose.position
        yaw = yaw_from_quat(self.pose.pose.orientation)
        pos = (p.x, p.y)
        pp = self.pp.compute(pos)

        # MPPI needs a goal roughly a horizon-distance ahead (not the short PP
        # look-ahead, which it would overshoot and oscillate around). Use a stable
        # PATH point ~mppi_goal_lookahead m ahead (path-based, not heading-based,
        # so it does not jitter).
        far_goal = self.pp.far_target(pos, self.mppi_goal_lookahead)

        if self.scan is None:
            target_dir = pp['heading']
            speed = min(self.mppi.vmax, max(0.0, 1.5 * pp['dist_to_goal']))
            vx, vy = speed * math.cos(target_dir), speed * math.sin(target_dir)
        else:
            res = self.mppi.compute(pos, yaw, list(self.scan.ranges),
                                    self.scan.angle_min, self.scan.angle_increment,
                                    far_goal)
            vx, vy = res['vx'], res['vy']
            cap = max(0.0, 1.5 * pp['dist_to_goal'])
            sp = math.hypot(vx, vy)
            if sp > cap and sp > 1e-6:
                vx, vy = vx * cap / sp, vy * cap / sp

        vz = clamp(self.kp_alt * (self.cruise_alt - p.z), -self.max_vz, self.max_vz)
        travel = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-3 else pp['heading']
        yaw_cmd = self._yaw_target(travel, dt)
        return self._make_vel_sp(vx, vy, vz, yaw_cmd), pp['finished']

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

        # --- VFH/FGM veto in body frame ---
        steer_world = desired_world
        clearance = float('inf')
        blocked = False
        if self.scan is not None:
            target_body = wrap_to_pi(desired_world - yaw)
            res = self.dir_planner.compute(list(self.scan.ranges),
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
            margin = self.dir_planner.robot_radius + self.dir_planner.safety_distance
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
                yaw = yaw_from_quat(self.pose.pose.orientation)
                if self.land_t0 is None:
                    self.land_t0 = rospy.Time.now()

                if self.land_auto and land_requested:
                    # AUTO.LAND engaged: keep nudging a setpoint in case we fall
                    # back to OFFBOARD, and watch for disarm / ground contact.
                    self.sp_pub.publish(
                        self._make_pos_sp(gx, gy, max(0.0, p.z - 0.5), self.yaw_cmd))
                    if (not self.state.armed) or p.z < 0.20:
                        rospy.loginfo('mission: landed and disarmed. DONE.')
                        break

                elif self.vision_landing and self._pad_fresh():
                    # ---- pixel-based precision alignment + descent ----
                    vx, vy = self._vision_servo(yaw)
                    centred = self.pad_err_norm < self.v_align_tol
                    # only sink once well-centred; otherwise hold altitude and slide
                    vz = -self.v_descend if centred else 0.0
                    self.sp_pub.publish(self._make_vel_sp(vx, vy, vz, 0.0))
                    rospy.loginfo_throttle(
                        1.0, 'mission: vision land  pix_err=%.3f (u=%.2f v=%.2f)  z=%.2f',
                        self.pad_err_norm, self.pad_err_u, self.pad_err_v, p.z)
                    # hand off to final touchdown once low and centred
                    if p.z <= self.v_handoff_alt and centred:
                        if self.land_auto:
                            if self._set_mode('AUTO.LAND'):
                                rospy.loginfo('mission: pad centred (pix_err=%.3f) '
                                              '-> AUTO.LAND', self.pad_err_norm)
                                land_requested = True
                        else:
                            self.phase = 'MANUAL_LAND'

                else:
                    # No camera fix (pad not visible yet, vision off, or it failed):
                    # hold over the known goal at cruise altitude and recentre.
                    self.sp_pub.publish(
                        self._make_pos_sp(gx, gy, self.cruise_alt, self.yaw_cmd))
                    dxy = math.hypot(p.x - gx, p.y - gy)
                    held = (rospy.Time.now() - self.land_t0).to_sec()
                    # Land on position alone when vision is disabled, or as a safety
                    # fallback if the pad was never acquired within the timeout.
                    use_pos_land = (not self.vision_landing) or held > self.v_fallback_after
                    if use_pos_land and dxy <= self.land_settle:
                        if self.vision_landing:
                            rospy.logwarn('mission: pad not acquired in %.1fs '
                                          '-> position-based landing', held)
                        if self.land_auto:
                            if self._set_mode('AUTO.LAND'):
                                rospy.loginfo('mission: centred over goal -> AUTO.LAND')
                                land_requested = True
                        else:
                            self.phase = 'MANUAL_LAND'

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

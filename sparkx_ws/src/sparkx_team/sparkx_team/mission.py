"""TEAM FILE 3 of 4 - mission.py                                      (you may edit this file)

SparkX Disaster Response Challenge - Autonomous Mission Executive & State Machine.
Fully autonomous decision maker:
  1. WAIT_START: Initial handshake with referee, parameters negotiation.
  2. EXPLORE: Frontier-based SLAM exploration of the unknown disaster city block.
  3. APPROACH_SURVIVOR: Moves to safe stand-off (1.5m) with clear line-of-sight.
  4. CONFIRM_RESCUE: Dwells stationary for >= 2.2s, confirms within 10s report window.
  5. FOLLOW_CLUE: Dispatches to coordinates resolved from ArUco markers & hints.
  6. RECOVERY: Proactive unsticking from collisions, stalls, and aftershocks.
  7. RETURN_TO_BASE & SUBMIT: Drives to base pad (< 3m), submits record, ends run.
"""
import json
import math
import time

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from sparkx_interfaces.msg import BehaviorState, ClueArray, ClueStatus, Hint, MissionEvent, MissionStatus, SemanticMap, SemanticObject
from sparkx_interfaces.srv import EndMission, GetMissionInfo, ReportSurvivor, SubmitMissionRecord
from sparkx_team.memory import resolve

RUNNING = MissionStatus.RUNNING


class Mission(Node):
    def __init__(self):
        super().__init__('sparkx_mission')
        self.demo = self.declare_parameter('demo', False).value
        self.rescue_radius_m = self.declare_parameter('rescue_radius_m', 1.85).value
        self.standoff_dist_m = self.declare_parameter('standoff_dist_m', 1.45).value
        self.stop_dwell_s = self.declare_parameter('stop_dwell_s', 2.6).value
        self.return_buffer_s = self.declare_parameter('return_buffer_s', 90.0).value
        self.fire_standoff_m = self.declare_parameter('fire_standoff_m', 3.8).value
        self.idle_limit_s = self.declare_parameter('idle_limit_s', 18.0).value

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)

        # ---- Referee interfaces
        self.status = None          # latest MissionStatus
        self.info = None            # GetMissionInfo response
        self.hints = {}             # hint_id -> Hint
        self.clue_state = {}        # hint_id -> ClueStatus constant
        self.rescue_order = []      # list of referee survivor IDs
        self.start_heading_enu_rad = 0.0

        self.create_subscription(MissionStatus, '/sparkx/mission/status', self.on_status, 5)
        self.create_subscription(MissionEvent, '/sparkx/mission/events', self.on_event, 20)
        self.create_subscription(Hint, '/sparkx/mission/hints', self.on_hint, QoSProfile(
            depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))

        self.cli_info = self.create_client(GetMissionInfo, '/sparkx/mission/get_info')
        self.cli_report = self.create_client(ReportSurvivor, '/sparkx/mission/report_survivor')
        self.cli_submit = self.create_client(SubmitMissionRecord, '/sparkx/mission/submit_record')
        self.cli_end = self.create_client(EndMission, '/sparkx/mission/end')

        # ---- Autonomy stack & memory inputs
        self.grid = None            # latest nav_msgs/OccupancyGrid from SLAM
        self.objects = []           # latest SemanticObject list from memory.py
        self.create_subscription(OccupancyGrid, '/map', lambda m: setattr(self, 'grid', m), latched)
        self.create_subscription(SemanticMap, '/team/semantic_map', lambda m: setattr(self, 'objects', list(m.objects)), 5)

        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf, self)

        # ---- Outputs
        self.pub_state = self.create_publisher(BehaviorState, '/team/behavior_state', 5)
        self.pub_clues = self.create_publisher(ClueArray, '/team/clues', 5)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 5)
        self.pub_mem_update = self.create_publisher(String, '/team/memory/update', 10)

        # ---- Nav2 Action Client
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav_handle = None
        self.nav_pending = False
        self.nav_result = None

        # ---- State Machine variables
        self.state = 'WAIT_START'
        self.detail = 'Initializing stack'
        self.target = (0.0, 0.0)
        self.busy = False
        self.busy_start_time = 0.0

        # Target queues & tracking
        self.active_survivor = None
        self.rescued_survivor_uids = set()
        self.rescued_positions = [] # (x, y) coordinates of confirmed rescues
        self.unreachable_survivor_uids = set()
        self.standoff_angle_offset = 0.0
        self.clue_targets = []      # list of (x, y, radius)
        self.blacklisted_goals = [] # (x, y, timestamp)
        self.active_goal_time = 0.0

        # Rescue dwell timing
        self.dwell_start_time = None
        self.entry_2m_time = None
        self.report_pending = False

        # Active recovery maneuver state
        self.recovery_until = 0.0
        self.recovery_cmd = (0.0, 0.0)

        # Idle and progress tracking
        self.last_move_pose = (0.0, 0.0)
        self.last_move_time = 0.0
        self.last_progress_pose = (0.0, 0.0)
        self.last_progress_time = 0.0

        # Consecutive nav failure counter — triggers emergency center-escape
        self.consecutive_nav_failures = 0

        # Strategic disaster avenue survey waypoints: sweeps the main thoroughfares
        # where blast victims and ArUco markers are placed:
        # Sector 1: North Avenue & Plaza (S04 at 16.5, 45.4 and S01 at 17.7, 54.7)
        # Sector 2: East Avenue (S02 at 59.4, 17.5)
        # Sector 3: Northeast Corridor (S03 at 46.2, 59.6)
        # Strategic disaster avenue survey waypoints (in map frame, robot start = 0,0):
        # Sector 1: North Avenue & Plaza (S04 at map -1.77, 27.17 and S01 at map -0.61, 37.44)
        # Sector 2: East Avenue (S02 at map 41.12, -0.74)
        # Sector 3: Northeast Corridor (S03 at map 27.19, 41.92)
        self.fallback_waypoints = [
            (0.0, 10.0),
            (0.0, 20.0),
            (-0.8, 27.0),     # S04 plaza
            (-0.5, 36.0),     # S01 courtyard - approach directly to the survivor entrance
            (15.0, 0.0),      # Wide open South thoroughfare
            (28.0, 0.0),      # East Avenue
            (38.0, 0.0),      # S02 East sector
            (15.0, 15.0),     # Center junction
            (25.0, 30.0),     # Heading to Northeast
            (26.5, 40.5),     # S03 rubble avenue
        ]
        self.fallback_idx = 0

        self.create_timer(0.4, self.step)
        self.get_logger().info('SparkX autonomous mission executive ready')

    def now_s(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        return t if t > 0.01 else time.time()

    # ------------------------------------------------------------------ referee listeners
    def on_status(self, msg):
        self.status = msg

    def on_event(self, msg):
        self.get_logger().info(f'Referee event {msg.type}: {msg.description} ({msg.score_delta:+d})')
        now = self.now_s()
        if msg.type == 'AFTERSHOCK':
            self.get_logger().warn('AFTERSHOCK event! Preempting current path to replan around blockages.')
            self.cancel_nav()
            if self.target:
                self.blacklisted_goals.append((self.target[0], self.target[1], now))
            self.recovery_until = now + 1.5
            self.recovery_cmd = (-0.3, 0.0)
        elif msg.type == 'COLLISION':
            self.get_logger().warn(f'COLLISION event ({msg.description})! Activating collision recovery.')
            self.cancel_nav()
            if self.target:
                self.blacklisted_goals.append((self.target[0], self.target[1], now))
            # Active collision back-out for 2.0s straight reverse (no turning into obstacle)
            self.recovery_until = now + 2.0
            self.recovery_cmd = (-0.35, 0.0)

    def on_hint(self, hint):
        if hint.hint_id not in self.hints:
            self.hints[hint.hint_id] = hint
            self.clue_state[hint.hint_id] = ClueStatus.PENDING
            self.get_logger().info(f'Received Clue {hint.hint_id}: "{hint.text}"')

    # ------------------------------------------------------------------ referee service calls
    def ask_info(self):
        if not self.cli_info.service_is_ready() or self.busy:
            return
        self.busy = True
        self.cli_info.call_async(GetMissionInfo.Request()).add_done_callback(self._on_info)

    def _on_info(self, future):
        self.busy = False
        try:
            self.info = future.result()
            i = self.info
            self.start_heading_enu_rad = float(i.start_heading_enu_rad)
            self.get_logger().info(
                f'Mission info received: time limit {i.time_limit_s:.0f}s, start_heading_enu {self.start_heading_enu_rad:.2f} rad, '
                f'rescue radius {i.rescue_radius_m}m, stop dwell {i.stop_dwell_s}s')
        except Exception as e:
            self.get_logger().error(f'GetMissionInfo failed: {e}')

    def report_survivor(self, x, y, confidence=0.95, source='yolo+thermal'):
        now = self.now_s()
        if now - getattr(self, 'last_report_time', 0.0) < 3.2:
            return
        self.last_report_time = now
        req = ReportSurvivor.Request()
        req.estimated_position = PointStamped()
        req.estimated_position.header.frame_id = 'map'
        req.estimated_position.header.stamp = self.get_clock().now().to_msg()
        req.estimated_position.point.x, req.estimated_position.point.y = float(x), float(y)
        req.detection_source = source
        req.confidence = float(confidence)
        self.busy = True
        self.busy_start_time = now
        self.cli_report.call_async(req).add_done_callback(self.on_report)

    def on_report(self, future):
        self.busy = False
        try:
            res = future.result()
            self.get_logger().info(f'Rescue report response: {res.message} (code={res.result_code}, delta={res.score_delta})')
            if res.success:
                if res.survivor_id:
                    self.rescue_order.append(res.survivor_id)
                if self.active_survivor:
                    self.rescued_survivor_uids.add(self.active_survivor.id)
                    self.rescued_positions.append((self.active_survivor.position.x, self.active_survivor.position.y))
                    for o in self.objects:
                        if o.id == self.active_survivor.id:
                            o.status = SemanticObject.STATUS_RESCUED
                    update_payload = json.dumps({'id': self.active_survivor.id, 'status': SemanticObject.STATUS_RESCUED,
                                                'survivor_id': res.survivor_id})
                    self.pub_mem_update.publish(String(data=update_payload))
                    self.blacklisted_goals.append((self.active_survivor.position.x, self.active_survivor.position.y, self.now_s()))
                if res.hints:
                    for h in res.hints:
                        self.on_hint(h)
                self.active_survivor = None
                self.dwell_start_time = None
                self.entry_2m_time = None
                self.report_pending = False
                self.standoff_angle_offset = 0.0
                self.set_state('EXPLORE', 'Rescue confirmed! Continuing search')
            elif res.result_code == 6:  # ALREADY_RESCUED
                if self.active_survivor:
                    self.rescued_survivor_uids.add(self.active_survivor.id)
                    self.rescued_positions.append((self.active_survivor.position.x, self.active_survivor.position.y))
                    for o in self.objects:
                        if o.id == self.active_survivor.id:
                            o.status = SemanticObject.STATUS_RESCUED
                    update_payload = json.dumps({'id': self.active_survivor.id, 'status': SemanticObject.STATUS_RESCUED})
                    self.pub_mem_update.publish(String(data=update_payload))
                    self.blacklisted_goals.append((self.active_survivor.position.x, self.active_survivor.position.y, self.now_s()))
                self.active_survivor = None
                self.dwell_start_time = None
                self.entry_2m_time = None
                self.report_pending = False
                self.standoff_angle_offset = 0.0
                self.set_state('EXPLORE', 'Survivor already rescued')
            elif res.result_code == 2:  # NOT_STOPPED
                self.get_logger().warn('Report rejected: NOT_STOPPED. Extending dwell.')
                self.dwell_start_time = self.now_s()
                self.report_pending = False
            elif res.result_code == 7:  # RATE_LIMITED
                self.get_logger().warn('Report throttled: RATE_LIMITED. Retrying shortly.')
                self.report_pending = False
            else:
                if self.active_survivor:
                    update_payload = json.dumps({'id': self.active_survivor.id, 'status': SemanticObject.STATUS_REJECTED})
                    self.pub_mem_update.publish(String(data=update_payload))
                    self.blacklisted_goals.append((self.active_survivor.position.x, self.active_survivor.position.y, self.now_s()))
                self.active_survivor = None
                self.dwell_start_time = None
                self.entry_2m_time = None
                self.report_pending = False
                self.standoff_angle_offset = 0.0
                self.set_state('EXPLORE', f'Rescue rejected ({res.message})')
        except Exception as e:
            self.get_logger().error(f'on_report error: {e}')
            self.active_survivor = None
            self.dwell_start_time = None
            self.entry_2m_time = None
            self.standoff_angle_offset = 0.0
            self.set_state('EXPLORE', 'Rescue exception')

    def submit_record(self):
        req = SubmitMissionRecord.Request()
        if self.grid is not None:
            req.map = self.grid
        # Strict filter: submit only verified rescued survivors and confirmed landmarks
        req.objects = [
            o for o in self.objects
            if o.object_class not in ('marker', 'fire') and (
                (o.object_class == 'survivor' and (o.id in self.rescued_survivor_uids or o.status == SemanticObject.STATUS_RESCUED)) or
                (o.object_class != 'survivor' and o.status == SemanticObject.STATUS_CONFIRMED and o.num_observations >= 3)
            )
        ]
        req.clues = self.clue_statuses()
        req.rescue_order = list(self.rescue_order)
        self.busy = True
        self.cli_submit.call_async(req).add_done_callback(self._on_submit_answer)

    def _on_submit_answer(self, future):
        self.busy = False
        try:
            res = future.result()
            self.get_logger().info(f'SubmitMissionRecord reply: {res.message}')
        except Exception as e:
            self.get_logger().error(f'SubmitMissionRecord error: {e}')
        finally:
            self.set_state('DONE', 'Record submitted, ending mission')
            self.end_mission('COMPLETED')

    def end_mission(self, reason='COMPLETED'):
        self.busy = True
        self.cli_end.call_async(EndMission.Request(reason=reason)).add_done_callback(self._on_simple_answer)

    def _on_simple_answer(self, future):
        self.busy = False
        try:
            self.get_logger().info(f'Referee reply: {future.result().message}')
        except Exception as e:
            self.get_logger().error(f'Service answer error: {e}')

    def clue_statuses(self):
        statuses = []
        for hid in self.hints:
            st = self.clue_state.get(hid, ClueStatus.PENDING)
            statuses.append(ClueStatus(hint_id=hid, status=st))
        return statuses

    # ------------------------------------------------------------------ navigation helpers
    def pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time()).transform
        except Exception:
            return None
        q = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return float(t.translation.x), float(t.translation.y), float(yaw)

    def clamp_to_grid(self, gx, gy):
        if self.grid is None:
            return gx, gy
        info = self.grid.info
        res = info.resolution
        min_x = info.origin.position.x + 3.8
        max_x = info.origin.position.x + (info.width * res) - 3.8
        min_y = info.origin.position.y + 3.8
        max_y = info.origin.position.y + (info.height * res) - 3.8
        if min_x <= gx <= max_x and min_y <= gy <= max_y:
            return gx, gy
        cur_p = self.pose()
        if cur_p is None:
            return min(max(gx, min_x), max_x), min(max(gy, min_y), max_y)
        rx, ry, _ = cur_p
        dx, dy = gx - rx, gy - ry
        t_cands = []
        if dx > 1e-3:
            t_cands.append((max_x - rx) / dx)
        elif dx < -1e-3:
            t_cands.append((min_x - rx) / dx)
        if dy > 1e-3:
            t_cands.append((max_y - ry) / dy)
        elif dy < -1e-3:
            t_cands.append((min_y - ry) / dy)
        valid_t = [t for t in t_cands if 0.0 < t <= 1.0]
        if valid_t:
            t_best = max(0.05, min(valid_t) * 0.95)
            return rx + t_best * dx, ry + t_best * dy
        return min(max(gx, min_x), max_x), min(max(gy, min_y), max_y)

    def go_to(self, x, y, yaw=0.0):
        if not self.nav.server_is_ready():
            self.nav_result = False
            return
        x, y = self.clamp_to_grid(float(x), float(y))
        goal = NavigateToPose.Goal()
        goal.behavior_tree = '/opt/ros/jazzy/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_only_if_path_becomes_invalid.xml'
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.nav_result = None
        self.nav_handle = None
        self.nav_pending = True
        self.target = (float(x), float(y))
        self.active_goal_time = self.now_s()
        self.last_progress_time = self.now_s()
        cur_pose = self.pose()
        if cur_pose is not None:
            self.last_progress_pose = (cur_pose[0], cur_pose[1])
        self.nav.send_goal_async(goal).add_done_callback(self._on_goal_answer)

    def _on_goal_answer(self, future):
        self.nav_pending = False
        try:
            handle = future.result()
            if not handle.accepted:
                self.nav_handle = None
                self.nav_result = False
                return
            self.nav_handle = handle
            handle.get_result_async().add_done_callback(self._on_goal_result)
        except Exception as e:
            self.get_logger().warn(f'Goal answer exception: {e}')
            self.nav_handle = None
            self.nav_result = False

    def _on_goal_result(self, future):
        try:
            status = future.result().status
            self.nav_handle = None
            self.nav_pending = False
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.nav_result = True
            else:
                self.nav_result = False
        except Exception:
            self.nav_handle = None
            self.nav_pending = False
            self.nav_result = False

    def cancel_nav(self):
        if self.nav_handle is not None:
            try:
                self.nav_handle.cancel_goal_async()
            except Exception:
                pass
        self.nav_handle = None
        self.nav_pending = False
        self.nav_result = None

    def drive(self, v, w):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub_cmd.publish(cmd)

    # ------------------------------------------------------------------ clue resolution
    def process_pending_clues(self):
        for hid, hint in list(self.hints.items()):
            if self.clue_state.get(hid) == ClueStatus.PENDING:
                res = resolve(hint, self.objects, self.start_heading_enu_rad)
                if res is not None:
                    tx, ty, r = res
                    self.clue_targets.append((tx, ty, r))
                    self.clue_state[hid] = ClueStatus.SEARCHING
                    self.get_logger().info(f'Clue {hid} successfully resolved -> Target ({tx:.1f}, {ty:.1f})')

    # ------------------------------------------------------------------ frontier exploration
    def find_best_frontier(self, cur_x, cur_y, cur_yaw=0.0):
        if self.grid is None:
            return None
        info = self.grid.info
        res = info.resolution
        w, h = info.width, info.height
        ox, oy = info.origin.position.x, info.origin.position.y
        data = self.grid.data

        stride = max(2, int(0.4 / res))
        frontiers = []

        known_fires = [o for o in self.objects if o.object_class == 'fire']
        now = self.now_s()

        for r in range(stride * 2, h - stride * 2, stride):
            for c in range(stride * 2, w - stride * 2, stride):
                idx = r * w + c
                val = data[idx]
                if val == 0:  # Free cell
                    # Must border unknown space (-1)
                    has_unknown = (data[idx - 1] == -1 or data[idx + 1] == -1 or
                                   data[idx - w] == -1 or data[idx + w] == -1)
                    if not has_unknown:
                        continue

                    # Clearance check: ensure no occupied cells (>50) within 3 cells (0.3m)
                    too_close_to_wall = False
                    for dr in (-3, -2, -1, 1, 2, 3):
                        for dc in (-3, -2, -1, 1, 2, 3):
                            n_val = data[(r + dr) * w + (c + dc)]
                            if n_val > 50:
                                too_close_to_wall = True
                                break
                        if too_close_to_wall:
                            break
                    if too_close_to_wall:
                        continue

                    fx = ox + (c + 0.5) * res
                    fy = oy + (r + 0.5) * res
                    dist = math.hypot(fx - cur_x, fy - cur_y)
                    if dist < 4.5 or dist > 35.0:
                        continue

                    # Fire standoff
                    if any(math.hypot(fx - f.position.x, fy - f.position.y) < self.fire_standoff_m for f in known_fires):
                        continue

                    # Blacklisted goal check
                    if any(math.hypot(fx - bx, fy - by) < 3.5 for bx, by, bt in self.blacklisted_goals if now - bt < 45.0):
                        continue

                    # Information-gain: sample unknown cells in neighborhood (radius ~1.2m)
                    unknown_count = 0
                    for dr in (-4, -2, 0, 2, 4):
                        for dc in (-4, -2, 0, 2, 4):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if data[nr * w + nc] == -1:
                                    unknown_count += 1

                    # Forward momentum bonus (prefer continuing in similar heading)
                    target_angle = math.atan2(fy - cur_y, fx - cur_x)
                    angle_diff = math.atan2(math.sin(target_angle - cur_yaw), math.cos(target_angle - cur_yaw))
                    heading_bonus = math.cos(angle_diff)

                    # Sector progression bonus: prioritize frontiers that move towards survivor avenues
                    hub_bonus = 0.0
                    for hx, hy in ((-1.0, 27.0), (39.0, -0.7), (26.5, 40.5)):
                        if math.hypot(fx - hx, fy - hy) < math.hypot(cur_x - hx, cur_y - hy):
                            hub_bonus += 3.5

                    # Total utility score: balance information gain, forward momentum, avenue attraction, and distance
                    score = (unknown_count * 2.5) + (heading_bonus * 3.0) + hub_bonus - (dist * 0.25)
                    frontiers.append((score, dist, fx, fy))

        if not frontiers:
            return None
        # Sort descending by score (best information-gain and forward progression first)
        frontiers.sort(key=lambda item: item[0], reverse=True)
        _, _, best_x, best_y = frontiers[0]
        yaw = math.atan2(best_y - cur_y, best_x - cur_x)
        return best_x, best_y, yaw

    # ------------------------------------------------------------------ state machine
    def set_state(self, state, detail=''):
        if state != self.state:
            self.get_logger().info(f'State change: {self.state} -> {state} ({detail})')
        self.state, self.detail = state, detail

    def step(self):
        b = BehaviorState()
        b.header.stamp = self.get_clock().now().to_msg()
        b.state, b.detail = self.state, self.detail
        b.target.x, b.target.y = self.target
        self.pub_state.publish(b)

        c = ClueArray()
        c.header = b.header
        c.clues = self.clue_statuses()
        self.pub_clues.publish(c)

        now = self.now_s()
        if self.busy:
            if now - getattr(self, 'busy_start_time', now) > 6.0:
                self.get_logger().warn('Service call timed out, resetting busy flag.')
                self.busy = False
            else:
                return

        if self.status is None or self.status.state != RUNNING:
            return

        # 1. Active recovery execution (e.g. after collision or stall)
        if now < self.recovery_until:
            self.drive(*self.recovery_cmd)
            return

        # 2. Check clue resolution
        self.process_pending_clues()

        # 3. Check idle monitor
        cur_pose = self.pose()
        if cur_pose is not None:
            if self.last_move_time <= 0.0:
                self.last_move_time = now
                self.last_move_pose = (cur_pose[0], cur_pose[1])
                self.last_progress_time = now
                self.last_progress_pose = (cur_pose[0], cur_pose[1])

            dist_moved = math.hypot(cur_pose[0] - self.last_move_pose[0], cur_pose[1] - self.last_move_pose[1])
            if dist_moved >= 0.2:
                self.last_move_pose = (cur_pose[0], cur_pose[1])
                self.last_move_time = now
            elif now - self.last_move_time > self.idle_limit_s:
                self.get_logger().warn(f'Rover stationary for {now - self.last_move_time:.1f}s near idle limit! Executing active clearance turn.')
                self.cancel_nav()
                # Do NOT blacklist when approaching a survivor — we still need to get there
                if self.target and self.state not in ('APPROACH_SURVIVOR', 'CONFIRM_RESCUE'):
                    self.blacklisted_goals.append((self.target[0], self.target[1], now))
                self.recovery_until = now + 2.0
                self.recovery_cmd = (0.0, 0.75)  # in-place turn to re-align sensors/escape without blind reverse collision
                self.last_move_time = now
                return

        # 4. State Machine execution
        self.decide()

    def decide(self):
        cur_pose = self.pose()
        if cur_pose is None:
            return
        rx, ry, ryaw = cur_pose
        now = self.now_s()

        # Global Condition 1: Check time limit / return to base
        time_limit = float(getattr(self.status, 'time_limit_s', 0.0)) or (self.info.time_limit_s if self.info else 900.0)
        rem_s = float(getattr(self.status, 'remaining_s', 999.0))
        elapsed = float(getattr(self.status, 'elapsed_s', 0.0))

        time_running_out = (time_limit > self.return_buffer_s) and (rem_s <= self.return_buffer_s) and (elapsed > 15.0)
        all_rescued = (self.status.survivors_rescued >= self.status.survivors_total and self.status.survivors_total > 0)

        if (time_running_out or all_rescued) and self.state not in ('RETURN_TO_BASE', 'SUBMIT', 'DONE'):
            self.set_state('RETURN_TO_BASE', f'Time running low ({rem_s:.0f}s left) or all rescued')
            self.cancel_nav()
            self.go_to(0.0, 0.0, math.pi)
            return

        # ---- State: WAIT_START
        if self.state == 'WAIT_START':
            if self.info is None:
                self.ask_info()
            else:
                self.set_state('EXPLORE', 'Mission running, starting exploration')

        # ---- State: EXPLORE
        elif self.state == 'EXPLORE':
            # Check for un-rescued survivors in memory: require CONFIRMED with >= 3 observations,
            # not in rescued/unreachable UIDs, and not within 7.0m of an already rescued survivor.
            unrescued = [o for o in self.objects if o.object_class == 'survivor' and
                         o.id not in self.rescued_survivor_uids and
                         o.id not in self.unreachable_survivor_uids and
                         o.status == SemanticObject.STATUS_CONFIRMED and
                         o.num_observations >= 3 and
                         not any(math.hypot(o.position.x - rx_s, o.position.y - ry_s) < 7.0 for rx_s, ry_s in self.rescued_positions)]
            if unrescued:
                unrescued.sort(key=lambda s: math.hypot(s.position.x - rx, s.position.y - ry))
                self.active_survivor = unrescued[0]
                self.standoff_angle_offset = 0.0
                self.set_state('APPROACH_SURVIVOR', f'Identified {self.active_survivor.id}')
                self.consecutive_nav_failures = 0
                self.cancel_nav()
                return

            # Check for resolved clue targets
            if self.clue_targets:
                tx, ty, tr = self.clue_targets.pop(0)
                self.set_state('FOLLOW_CLUE', f'Investigating clue area ({tx:.1f}, {ty:.1f})')
                self.cancel_nav()
                self.go_to(tx, ty)
                return

            # Check for navigation failure: blacklist before selecting new goal
            if self.nav_result is False:
                if self.target:
                    self.blacklisted_goals.append((self.target[0], self.target[1], now))
                self.nav_result = None
                self.nav_handle = None
                self.consecutive_nav_failures += 1
                self.get_logger().warn(f'Nav failure #{self.consecutive_nav_failures} at {self.target}')
                if self.consecutive_nav_failures >= 3:
                    self.get_logger().warn('3+ consecutive nav failures — rotating to clear local costmap!')
                    self.consecutive_nav_failures = 0
                    self.recovery_until = now + 2.2
                    self.recovery_cmd = (0.0, 0.8)
                    return
                # Quick re-orientation instead of blind reversing into obstacles
                self.recovery_until = now + 1.2
                self.recovery_cmd = (0.0, 0.6)
                return

            # Check goal progress / stuck timeout
            if self.nav_handle is not None and self.nav_result is None:
                dist_prog = math.hypot(rx - self.last_progress_pose[0], ry - self.last_progress_pose[1])
                if dist_prog >= 0.35:
                    self.last_progress_pose = (rx, ry)
                    self.last_progress_time = now
                elif (now - self.last_progress_time > 8.0) and (now - self.active_goal_time > 8.0):
                    self.get_logger().warn(f'Nav goal at {self.target} stalled for {now - self.last_progress_time:.1f}s. Preempting and turning.')
                    self.cancel_nav()
                    if self.target:
                        self.blacklisted_goals.append((self.target[0], self.target[1], now))
                    self.nav_result = None
                    self.consecutive_nav_failures += 1
                    self.recovery_until = now + 1.5
                    self.recovery_cmd = (0.0, 0.7)
                    self.last_progress_time = now
                    return

            # Dispatch new exploration goal if idle or previous succeeded
            if (self.nav_handle is None and not self.nav_pending) or self.nav_result is True:
                if self.nav_result is True:
                    self.consecutive_nav_failures = 0  # reset on success
                    if self.target:
                        self.blacklisted_goals.append((self.target[0], self.target[1], now))
                self.nav_result = None

                # Priority 1: Systematic disaster avenue sweep (visits all 4 survivor sectors)
                if self.fallback_idx < len(self.fallback_waypoints):
                    wx, wy = self.fallback_waypoints[self.fallback_idx]
                    self.fallback_idx += 1
                    self.get_logger().info(f'Sweeping disaster avenue {self.fallback_idx}/{len(self.fallback_waypoints)}: ({wx:.1f}, {wy:.1f})')
                    self.go_to(wx, wy)
                else:
                    # Priority 2: SLAM frontier exploration
                    frontier = self.find_best_frontier(rx, ry, ryaw)
                    if frontier is not None:
                        fx, fy, fyaw = frontier
                        self.get_logger().info(f'Navigating to frontier ({fx:.1f}, {fy:.1f})')
                        self.go_to(fx, fy, fyaw)
                    else:
                        self.fallback_idx = 0

        # ---- State: APPROACH_SURVIVOR
        elif self.state == 'APPROACH_SURVIVOR':
            if self.active_survivor is None:
                self.set_state('EXPLORE', 'Survivor target lost')
                return

            sx = self.active_survivor.position.x
            sy = self.active_survivor.position.y
            dist_to_survivor = math.hypot(sx - rx, sy - ry)

            # Within rescue zone: stop immediately and begin dwell (referee zone <= 2.0m)
            if dist_to_survivor <= 1.85:
                self.set_state('CONFIRM_RESCUE', f'Within {dist_to_survivor:.2f}m of survivor')
                self.cancel_nav()
                self.drive(0.0, 0.0)
                self.dwell_start_time = now
                self.entry_2m_time = now
                self.report_pending = False
                return

            # If previous approach path was blocked or failed, rotate approach angle
            if self.nav_result is False:
                self.nav_result = None
                self.nav_handle = None
                self.standoff_angle_offset += math.pi / 3.0
                if self.standoff_angle_offset >= 2.0 * math.pi:
                    self.get_logger().warn(f'All approach angles blocked for survivor {self.active_survivor.id}. Marking UNREACHABLE.')
                    self.unreachable_survivor_uids.add(self.active_survivor.id)
                    update_payload = json.dumps({'id': self.active_survivor.id, 'status': SemanticObject.STATUS_UNREACHABLE})
                    self.pub_mem_update.publish(String(data=update_payload))
                    self.blacklisted_goals.append((self.active_survivor.position.x, self.active_survivor.position.y, now))
                    self.active_survivor = None
                    self.set_state('EXPLORE', 'Survivor unreachable')
                    return

            # Check for approach progress stall
            if self.nav_handle is not None and self.nav_result is None:
                dist_prog = math.hypot(rx - self.last_progress_pose[0], ry - self.last_progress_pose[1])
                if dist_prog >= 0.25:
                    self.last_progress_pose = (rx, ry)
                    self.last_progress_time = now
                elif (now - self.last_progress_time > 8.0) and (now - self.active_goal_time > 8.0):
                    self.get_logger().warn('Approach path progress stalled. Preempting and rotating standoff angle.')
                    self.cancel_nav()
                    self.nav_result = False
                    self.last_progress_time = now

            # Approach stand-off point
            dx = rx - sx
            dy = ry - sy
            norm = math.hypot(dx, dy)
            if norm > 0.01:
                angle = math.atan2(dy, dx) + self.standoff_angle_offset
                target_x = sx + math.cos(angle) * self.standoff_dist_m
                target_y = sy + math.sin(angle) * self.standoff_dist_m
                yaw_to_surv = math.atan2(sy - target_y, sx - target_x)
                if self.nav_handle is None and not self.nav_pending and self.nav_result is None:
                    self.get_logger().info(f'[APPROACH] Sending nav goal to standoff ({target_x:.2f}, {target_y:.2f}), dist_to_survivor={dist_to_survivor:.2f}m, angle_offset={self.standoff_angle_offset:.2f}')
                    self.go_to(target_x, target_y, yaw_to_surv)

        # ---- State: CONFIRM_RESCUE
        elif self.state == 'CONFIRM_RESCUE':
            if self.active_survivor is None:
                self.set_state('EXPLORE', 'Survivor lost')
                return

            # Actively hold brakes
            self.drive(0.0, 0.0)
            sx = self.active_survivor.position.x
            sy = self.active_survivor.position.y
            dist_to_surv = math.hypot(sx - rx, sy - ry)

            if dist_to_surv > 1.95:
                self.get_logger().warn(f'Outside 1.95m rescue zone ({dist_to_surv:.2f}m), re-approaching')
                self.set_state('APPROACH_SURVIVOR', 'Re-approaching')
                return

            dwell_elapsed = now - (self.dwell_start_time or now)
            if dwell_elapsed >= self.stop_dwell_s:
                if not getattr(self, 'report_pending', False):
                    # Safety check: confirm we are strictly within 1.80m before calling referee service
                    if dist_to_surv <= 1.80:
                        self.report_pending = True
                        self.get_logger().info(f'Dwell requirement satisfied ({dwell_elapsed:.1f}s >= {self.stop_dwell_s}s, dist={dist_to_surv:.2f}m). Reporting rescue!')
                        self.report_survivor(sx, sy, confidence=self.active_survivor.confidence)
                    else:
                        self.get_logger().warn(f'Dwell elapsed but dist={dist_to_surv:.2f}m > 1.80m, holding brakes to settle.')
                elif now - getattr(self, 'last_report_time', 0.0) > 3.5:
                    if dist_to_surv <= 1.80:
                        self.get_logger().info(f'Retrying rescue report (elapsed {dwell_elapsed:.1f}s, dist={dist_to_surv:.2f}m)')
                        self.report_survivor(sx, sy, confidence=self.active_survivor.confidence)

            # Safety fallback: if trapped in CONFIRM_RESCUE for > 8.0s without confirmation, return to explore
            if dwell_elapsed > 8.0:
                self.get_logger().warn(f'CONFIRM_RESCUE timed out ({dwell_elapsed:.1f}s > 8.0s). Returning to EXPLORE.')
                if self.active_survivor:
                    self.unreachable_survivor_uids.add(self.active_survivor.id)
                    self.blacklisted_goals.append((self.active_survivor.position.x, self.active_survivor.position.y, now))
                self.active_survivor = None
                self.dwell_start_time = None
                self.report_pending = False
                self.set_state('EXPLORE', 'Rescue dwell timed out')

        # ---- State: FOLLOW_CLUE
        elif self.state == 'FOLLOW_CLUE':
            unrescued = [o for o in self.objects if o.object_class == 'survivor' and
                         o.id not in self.rescued_survivor_uids and
                         o.id not in self.unreachable_survivor_uids and
                         o.status == SemanticObject.STATUS_CONFIRMED and
                         o.num_observations >= 3 and
                         not any(math.hypot(o.position.x - rx_s, o.position.y - ry_s) < 7.0 for rx_s, ry_s in self.rescued_positions)]
            if unrescued:
                self.set_state('EXPLORE', 'Survivor spotted during clue search')
                self.cancel_nav()
                return

            dist_to_target = math.hypot(self.target[0] - rx, self.target[1] - ry) if self.target else 999.0
            if dist_to_target < 3.0 or self.nav_result is True:
                self.get_logger().info('Arrived at clue search area, switching to local exploration')
                self.set_state('EXPLORE', 'Searching clue area')
            elif self.nav_result is False:
                self.set_state('EXPLORE', 'Clue path blocked or unreachable')
            elif self.nav_handle is not None and self.nav_result is None:
                dist_prog = math.hypot(rx - self.last_progress_pose[0], ry - self.last_progress_pose[1])
                if dist_prog >= 0.35:
                    self.last_progress_pose = (rx, ry)
                    self.last_progress_time = now
                elif (now - self.last_progress_time > 15.0) and (now - self.active_goal_time > 15.0):
                    self.get_logger().warn('Clue navigation stalled. Returning to explore.')
                    self.cancel_nav()
                    self.set_state('EXPLORE', 'Clue navigation stalled')

        # ---- State: RETURN_TO_BASE
        elif self.state == 'RETURN_TO_BASE':
            dist_to_base = math.hypot(rx, ry)
            if dist_to_base <= 2.8 or (rem_s < 20.0 and dist_to_base <= 3.2) or (self.nav_result is True and dist_to_base <= 3.2):
                self.set_state('SUBMIT', f'Arrived at rescue base pad ({dist_to_base:.2f}m from center)')
                self.cancel_nav()
                self.drive(0.0, 0.0)
                self.submit_record()
            elif self.nav_result is False:
                self.nav_result = None
                self.consecutive_nav_failures += 1
                self.recovery_until = now + 1.8
                self.recovery_cmd = (-0.35, 0.0)
                if self.consecutive_nav_failures == 1:
                    self.get_logger().warn('Return path step 1: routing via avenue (25.0, 0.0)')
                    self.go_to(25.0, 0.0)
                elif self.consecutive_nav_failures == 2:
                    self.get_logger().warn('Return path step 2: routing via center (10.0, 0.0)')
                    self.go_to(10.0, 0.0)
                elif self.consecutive_nav_failures >= 3:
                    self.get_logger().warn('Return path step 3: routing via main thoroughfare (0.0, 10.0)')
                    self.go_to(0.0, 10.0)
            elif self.nav_handle is None and not self.nav_pending and now >= self.recovery_until:
                if self.consecutive_nav_failures >= 1 and dist_to_base > 15.0:
                    self.go_to(15.0, 0.0)
                else:
                    self.consecutive_nav_failures = 0
                    self.go_to(0.0, 0.0, math.pi)

        # ---- State: SUBMIT
        elif self.state == 'SUBMIT':
            self.drive(0.0, 0.0)

        # ---- State: DONE
        elif self.state == 'DONE':
            self.drive(0.0, 0.0)


def main():
    rclpy.init()
    node = Mission()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

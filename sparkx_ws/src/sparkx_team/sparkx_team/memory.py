"""TEAM FILE 2 of 4 - memory.py                                       (you may edit this file)

SparkX Disaster Response Challenge - Spatial Semantic Memory & Clue Resolution.
Responsibilities:
  1. Spatial clustering & track fusion of 3D detections into persistent objects in "map" frame.
  2. Clue resolution (resolve): Converts referee clues into concrete search coordinates (x, y, radius).
  3. Keepout publisher: Generates obstacle point cloud on /team/keepout for Nav2 costmaps
     (keeping rover >= 3.4m from fires and preventing collision with survivors).
  4. Publishes /team/semantic_map for competition scoring and mission decisions.
"""
import json
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, String
from vision_msgs.msg import Detection3DArray

from sparkx_interfaces.msg import SemanticMap, SemanticObject

MAX_OBJECTS = 200


def distance_pts(p1, p2):
    """Horizontal distance between two points (with .x, .y)."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def distance_coords(x1, y1, x2, y2):
    """Horizontal distance between two (x, y) tuples."""
    return math.hypot(x1 - x2, y1 - y2)


def resolve(hint, objects, start_heading_enu_rad):
    """Turn a clue into a search area on your map: return (x, y, radius_m), or None if it cannot be
    resolved yet (for example because the marker it names has not been seen).

    `hint` is a sparkx_interfaces/Hint, `objects` the SemanticObject list of this node,
    `start_heading_enu_rad` comes from GetMissionInfo.

    The marker part of a clue: the person is `hint.marker_distance_m` metres from marker
    `hint.marker_id`, on compass bearing `hint.marker_bearing_deg` (0 = north, 90 = east).
    With a = pi/2 - radians(bearing) - start_heading_enu_rad, the point in your map frame is
        marker_position + distance * (cos a, sin a)
    and the person is within 2.5 m of it (docs/INTERFACES.md, section 4)."""
    if hint is None:
        return None

    # Priority 1: Numbered location marker reference (precise to 2.5m)
    if hint.marker_id >= 0:
        target_tag = str(hint.marker_id)
        marker_obj = next((o for o in objects if o.object_class == 'marker' and str(o.label) == target_tag), None)
        if marker_obj is not None:
            mx = marker_obj.position.x
            my = marker_obj.position.y
            bearing_rad = math.radians(hint.marker_bearing_deg)
            # Convert compass bearing to map frame angle
            a = (math.pi / 2.0) - bearing_rad - start_heading_enu_rad
            tx = mx + hint.marker_distance_m * math.cos(a)
            ty = my + hint.marker_distance_m * math.sin(a)
            return (float(tx), float(ty), 2.5)

    # Priority 2: Named Landmark reference (approximate)
    if hasattr(hint, 'landmark') and hint.landmark.landmark_class:
        lm_class = hint.landmark.landmark_class
        lm_color = hint.landmark.color
        lm_label = hint.landmark.label

        lm_obj = next((o for o in objects if o.object_class == lm_class and
                       (not lm_color or o.color == lm_color) and
                       (not lm_label or o.label == lm_label)), None)
        if lm_obj is not None:
            lx, ly = lm_obj.position.x, lm_obj.position.y
            dist = hint.approx_distance_m if hint.approx_distance_m > 0 else 5.0
            rel = hint.relation.lower() if hint.relation else 'near'

            if rel in ('near', 'beside', 'inside'):
                return (float(lx), float(ly), 4.5)
            elif rel == 'north_of':
                ang = (math.pi / 2.0) - start_heading_enu_rad
                return (float(lx + dist * math.cos(ang)), float(ly + dist * math.sin(ang)), 4.0)
            elif rel == 'south_of':
                ang = (-math.pi / 2.0) - start_heading_enu_rad
                return (float(lx + dist * math.cos(ang)), float(ly + dist * math.sin(ang)), 4.0)
            elif rel == 'east_of':
                ang = 0.0 - start_heading_enu_rad
                return (float(lx + dist * math.cos(ang)), float(ly + dist * math.sin(ang)), 4.0)
            elif rel == 'west_of':
                ang = math.pi - start_heading_enu_rad
                return (float(lx + dist * math.cos(ang)), float(ly + dist * math.sin(ang)), 4.0)
            elif rel in ('behind', 'opposite'):
                return (float(lx), float(ly), 7.0)
            else:
                return (float(lx), float(ly), 5.0)

    return None


class TrackedEntity:
    def __init__(self, uid, obj_class, color, label, pos, conf):
        self.id = uid
        self.object_class = obj_class
        self.color = color
        self.label = str(label)
        self.x = float(pos.x)
        self.y = float(pos.y)
        self.z = float(pos.z)
        self.confidence = float(conf)
        self.num_observations = 1
        self.status = SemanticObject.STATUS_TENTATIVE
        self.survivor_id = ''

    def update(self, pos, conf):
        # Weighted running average
        w = 1.0 / (self.num_observations + 1)
        self.x = (1.0 - w) * self.x + w * float(pos.x)
        self.y = (1.0 - w) * self.y + w * float(pos.y)
        self.z = (1.0 - w) * self.z + w * float(pos.z)
        self.confidence = max(self.confidence, float(conf))
        self.num_observations += 1
        if self.num_observations >= 2 and self.status == SemanticObject.STATUS_TENTATIVE:
            self.status = SemanticObject.STATUS_CONFIRMED

    def to_msg(self):
        o = SemanticObject()
        o.id = self.id
        o.object_class = self.object_class
        o.color = self.color
        o.label = self.label
        o.position.x = float(self.x)
        o.position.y = float(self.y)
        o.position.z = float(self.z)
        o.confidence = float(self.confidence)
        o.num_observations = int(self.num_observations)
        o.status = self.status
        o.survivor_id = self.survivor_id
        return o


class SemanticMemory(Node):
    def __init__(self):
        super().__init__('sparkx_semantic_memory')
        self.survivors = {}       # id -> TrackedEntity
        self.markers = {}         # tag_label -> TrackedEntity
        self.fires = {}           # id -> TrackedEntity
        self.landmarks = {}       # id -> TrackedEntity

        self.survivor_seq = 0
        self.fire_seq = 0
        self.landmark_seq = 0

        self.create_subscription(Detection3DArray, '/perception/detections', self.on_detections, 20)
        self.create_subscription(String, '/team/memory/update', self.on_update, 20)

        self.pub_sem = self.create_publisher(SemanticMap, '/team/semantic_map', 10)
        self.pub_keepout = self.create_publisher(PointCloud2, '/team/keepout', 5)

        self.publish_rate_hz = self.declare_parameter('publish_rate_hz', 2.0).value
        self.create_timer(1.0 / self.publish_rate_hz, self.publish_all)
        self.get_logger().info('SparkX spatial memory node ready (track fusion & keepout active)')

    def on_detections(self, msg):
        """Fuses 3D detections into persistent tracked entities."""
        for det in msg.detections:
            if not det.results:
                continue
            h = det.results[0]
            cid = h.hypothesis.class_id
            cls, _, rest = cid.partition(':')
            pos = h.pose.pose.position
            conf = float(h.hypothesis.score)

            if cls == 'marker':
                tag_label = rest.strip()
                if not tag_label:
                    continue
                if tag_label in self.markers:
                    self.markers[tag_label].update(pos, conf)
                else:
                    uid = f'marker_{tag_label}'
                    self.markers[tag_label] = TrackedEntity(uid, 'marker', '', tag_label, pos, conf)

            elif cls == 'fire':
                matched = False
                for f in self.fires.values():
                    if distance_coords(f.x, f.y, pos.x, pos.y) <= 3.5:
                        f.update(pos, conf)
                        matched = True
                        break
                if not matched:
                    self.fire_seq += 1
                    uid = f'fire_{self.fire_seq}'
                    self.fires[uid] = TrackedEntity(uid, 'fire', '', '', pos, conf)

            elif cls == 'survivor':
                matched = False
                # Rules: Survivors are never closer than 7m. Matching distance 6.5m prevents split duplicates.
                for s in self.survivors.values():
                    if distance_coords(s.x, s.y, pos.x, pos.y) <= 6.5:
                        matched = True
                        # If previously rejected or already rescued by referee, do NOT re-arm
                        if s.status not in (SemanticObject.STATUS_REJECTED, SemanticObject.STATUS_RESCUED):
                            s.update(pos, conf)
                        break
                if not matched:
                    self.survivor_seq += 1
                    uid = f'survivor_{self.survivor_seq}'
                    self.survivors[uid] = TrackedEntity(uid, 'survivor', '', '', pos, conf)

            else:
                # Other landmarks: container, tent, water_tank, sign, car, etc.
                matched = False
                for lm in self.landmarks.values():
                    if lm.object_class == cls and distance_coords(lm.x, lm.y, pos.x, pos.y) <= 4.0:
                        lm.update(pos, conf)
                        matched = True
                        break
                if not matched:
                    self.landmark_seq += 1
                    uid = f'{cls}_{self.landmark_seq}'
                    label = rest if cls == 'sign' else ''
                    color = rest if cls != 'sign' else ''
                    self.landmarks[uid] = TrackedEntity(uid, cls, color, label, pos, conf)

    def on_update(self, msg):
        """mission.py reports what became of an object: {"id": "survivor_3", "status": "RESCUED", "survivor_id": "S02"}"""
        try:
            u = json.loads(msg.data)
        except Exception:
            return
        uid = u.get('id')
        new_status = u.get('status')
        new_survivor_id = u.get('survivor_id')

        for store in (self.survivors, self.markers, self.fires, self.landmarks):
            for k, obj in list(store.items()):
                if obj.id == uid:
                    if new_status:
                        obj.status = new_status
                    if new_survivor_id:
                        obj.survivor_id = new_survivor_id

    def publish_all(self):
        now = self.get_clock().now().to_msg()

        # 1. Build SemanticMap (only confirmed or rescued objects to avoid false object penalties)
        all_objects = []
        for s in self.survivors.values():
            if s.status in (SemanticObject.STATUS_CONFIRMED, SemanticObject.STATUS_RESCUED):
                all_objects.append(s.to_msg())
        for m in self.markers.values():
            all_objects.append(m.to_msg())
        for lm in self.landmarks.values():
            if lm.num_observations >= 3 or lm.confidence >= 0.85:
                all_objects.append(lm.to_msg())

        sem_map = SemanticMap()
        sem_map.header.frame_id = 'map'
        sem_map.header.stamp = now
        sem_map.objects = all_objects
        self.pub_sem.publish(sem_map)

        # 2. Build Keepout PointCloud2 for Nav2 Costmaps
        keepout_pts = []
        for f in self.fires.values():
            # Concentric rings up to 3.6m (fire standoff requirement >= 3.4m)
            for r in [0.8, 1.6, 2.4, 3.2, 3.6]:
                n_pts = max(8, int(r * 8))
                for i in range(n_pts):
                    theta = (2.0 * math.pi * i) / n_pts
                    keepout_pts.append([f.x + r * math.cos(theta), f.y + r * math.sin(theta), 0.4])

        for s in self.survivors.values():
            if s.status != SemanticObject.STATUS_REJECTED:
                # 0.45m stand-off ring so rover does not touch survivor (-50 pt penalty)
                for r in [0.0, 0.45]:
                    if r == 0.0:
                        keepout_pts.append([s.x, s.y, 0.3])
                    else:
                        for i in range(8):
                            theta = (2.0 * math.pi * i) / 8
                            keepout_pts.append([s.x + r * math.cos(theta), s.y + r * math.sin(theta), 0.3])

        if keepout_pts:
            hdr = Header()
            hdr.frame_id = 'map'
            hdr.stamp = now
            cloud = pc2.create_cloud_xyz32(hdr, keepout_pts)
            self.pub_keepout.publish(cloud)


def main():
    rclpy.init()
    node = SemanticMemory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

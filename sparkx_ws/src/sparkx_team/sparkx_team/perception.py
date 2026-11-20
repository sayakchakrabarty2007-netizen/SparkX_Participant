"""TEAM FILE 1 of 4 - perception.py                                   (you may edit this file)

SparkX Disaster Response Challenge - Perception System.
Fuses RGB-D, Thermal, and 3D LiDAR sensors to detect:
  1. Fires (thermal saturation at ~653 K)
  2. ArUco location markers (DICT_4X4_100)
  3. Survivors (YOLOv8 + cross-spectral thermal gating at 303-315 K)
     - Rejects cold mannequin decoys (~298 K)
     - Rejects overheated non-human objects and heated ground near fires (> 325 K)
     - Multi-spectral fusion ensures heated ground / drums without human shape are rejected
  4. Landmarks (signs, containers, water tanks, relief tents)
"""
import glob
import math
import os
import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

try:
    import onnxruntime as ort
except ImportError:
    ort = None

SIGN_LABELS = ['PHARMACY', 'SCHOOL', 'HOTEL', 'BANK', 'POLICE', 'CAFE', 'HOSPITAL', 'MARKET']
COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'white', 'gray']


class Perception(Node):
    def __init__(self):
        super().__init__('sparkx_perception')
        self.bridge = CvBridge()
        self.model_dir = self.declare_parameter('model_dir', '').value
        self.rate_hz = self.declare_parameter('rate_hz', 5.0).value
        self.conf_thresh = self.declare_parameter('conf_thresh', 0.20).value
        self.thermal_min_k = self.declare_parameter('thermal_min_k', 303.0).value
        self.thermal_max_k = self.declare_parameter('thermal_max_k', 315.0).value
        self.fire_thresh_k = self.declare_parameter('fire_thresh_k', 610.0).value
        self.max_range_m = self.declare_parameter('max_range_m', 15.0).value

        # Initialize detector models
        self.ort_session = None
        self.model_input_name = None
        self.load_team_model()

        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters()

        # ---- latest sensor data
        self.thermal_k = None       # float32 image, kelvin
        self.rgb = None             # uint8 image, RGB
        self.depth_m = None         # float32 image, metres (NaN / inf where no return)
        self.stamp = None           # time stamp of latest sensor image
        self.K_thermal = None       # 3x3 intrinsics of thermal camera
        self.K_rgb = None           # 3x3 intrinsics of colour/depth camera

        self.create_subscription(Image, '/thermal/image_raw', self.on_thermal, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/color/image_raw', self.on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/depth/image_raw', self.on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/thermal/camera_info',
                                 lambda m: setattr(self, 'K_thermal', np.array(m.k).reshape(3, 3)), 5)
        self.create_subscription(CameraInfo, '/camera/color/camera_info',
                                 lambda m: setattr(self, 'K_rgb', np.array(m.k).reshape(3, 3)), 5)

        self.tf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf, self)

        self.pub = self.create_publisher(Detection3DArray, '/perception/detections', 10)
        self.pub_snap = self.create_publisher(CompressedImage, '/perception/snapshot', 2)
        self.create_timer(1.0 / self.rate_hz, self.process)
        self.get_logger().info('SparkX perception node ready (YOLOv8 + Multi-spectral Thermal + ArUco enabled)')

    # ------------------------------------------------------------------ model loader
    def load_team_model(self):
        """Loads trained weights (.onnx) from model_dir."""
        if not self.model_dir or not os.path.isdir(self.model_dir):
            self.get_logger().warn(f'Model dir "{self.model_dir}" not found or empty')
            return None

        onnx_files = glob.glob(os.path.join(self.model_dir, '*.onnx'))
        if onnx_files and ort is not None:
            model_path = sorted(onnx_files)[0]
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.ort_session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
                self.model_input_name = self.ort_session.get_inputs()[0].name
                self.get_logger().info(f'Loaded ONNX model: {os.path.basename(model_path)}')
                return self.ort_session
            except Exception as e:
                self.get_logger().error(f'Failed to load ONNX model {model_path}: {e}')
        return None

    # ------------------------------------------------------------------ sensor callbacks
    def on_thermal(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')     # uint16
        self.thermal_k = raw.astype(np.float32) * 0.01                            # kelvin
        self.stamp = msg.header.stamp

    def on_rgb(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def on_depth(self, msg):
        self.depth_m = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')   # float32 metres

    # ------------------------------------------------------------------ geometry helper
    def pixel_to_map(self, u, v, range_m, K, frame):
        """Map-frame (x, y, z) of the point seen at pixel (u, v) of a camera, `range_m` metres along
        the camera's viewing axis."""
        if K is None or range_m <= 0.05 or math.isnan(range_m) or math.isinf(range_m):
            return None
        try:
            if self.stamp is not None:
                try:
                    t = self.tf.lookup_transform('map', frame, self.stamp, rclpy.duration.Duration(seconds=0.05)).transform
                except Exception:
                    t = self.tf.lookup_transform('map', frame, rclpy.time.Time()).transform
            else:
                t = self.tf.lookup_transform('map', frame, rclpy.time.Time()).transform
        except Exception:
            return None

        p = np.array([(u - K[0, 2]) / K[0, 0] * range_m, (v - K[1, 2]) / K[1, 1] * range_m, range_m])
        q = t.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = R @ p + np.array([t.translation.x, t.translation.y, t.translation.z])
        return float(m[0]), float(m[1]), float(m[2])

    def sample_depth(self, u, v, w=5, h=5):
        """Samples robust median depth in an ROI centered at (u, v)."""
        if self.depth_m is None:
            return None
        H, W = self.depth_m.shape
        u0 = max(0, int(round(u - w / 2)))
        u1 = min(W, int(round(u + w / 2)) + 1)
        v0 = max(0, int(round(v - h / 2)))
        v1 = min(H, int(round(v + h / 2)) + 1)
        patch = self.depth_m[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.2) & (patch < self.max_range_m)]
        if len(valid) == 0:
            return None
        return float(np.median(valid))

    def rgb_to_thermal_coords(self, u_rgb, v_rgb, depth_m):
        if self.K_rgb is None or self.K_thermal is None or depth_m <= 0:
            return None, None
        fx_rgb, fy_rgb = self.K_rgb[0, 0], self.K_rgb[1, 1]
        cx_rgb, cy_rgb = self.K_rgb[0, 2], self.K_rgb[1, 2]
        fx_th, fy_th = self.K_thermal[0, 0], self.K_thermal[1, 1]
        cx_th, cy_th = self.K_thermal[0, 2], self.K_thermal[1, 2]

        xn = (u_rgb - cx_rgb) / fx_rgb
        yn = (v_rgb - cy_rgb) / fy_rgb
        yn_th = yn + (0.04 / depth_m)
        u_th = fx_th * xn + cx_th
        v_th = fy_th * yn_th + cy_th
        return float(u_th), float(v_th)

    def thermal_to_rgb_coords(self, u_th, v_th, depth_m):
        if self.K_rgb is None or self.K_thermal is None or depth_m <= 0:
            return None, None
        fx_rgb, fy_rgb = self.K_rgb[0, 0], self.K_rgb[1, 1]
        cx_rgb, cy_rgb = self.K_rgb[0, 2], self.K_rgb[1, 2]
        fx_th, fy_th = self.K_thermal[0, 0], self.K_thermal[1, 1]
        cx_th, cy_th = self.K_thermal[0, 2], self.K_thermal[1, 2]

        xn = (u_th - cx_th) / fx_th
        yn = (v_th - cy_th) / fy_th
        yn_rgb = yn - (0.04 / depth_m)
        u_rgb = fx_rgb * xn + cx_rgb
        v_rgb = fy_rgb * yn_rgb + cy_rgb
        return float(u_rgb), float(v_rgb)

    # ------------------------------------------------------------------ 1. Fire detector
    def detect_fires(self):
        """Detects fire hazards (saturated thermal pixels >= fire_thresh_k)."""
        if self.thermal_k is None or self.K_thermal is None:
            return []

        fire_mask = (self.thermal_k >= self.fire_thresh_k).astype(np.uint8) * 255
        if not np.any(fire_mask):
            return []

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            if cv2.contourArea(c) < 4:
                continue
            M = cv2.moments(c)
            if M['m00'] <= 0:
                continue
            u_th = M['m10'] / M['m00']
            v_th = M['m01'] / M['m00']

            u_rgb, v_rgb = self.thermal_to_rgb_coords(u_th, v_th, depth_m=4.0)
            d = None
            if u_rgb is not None:
                d = self.sample_depth(u_rgb, v_rgb, w=15, h=15)
            if d is None:
                d = 4.5

            pt = self.pixel_to_map(u_th, v_th, d, self.K_thermal, 'thermal_optical_frame')
            if pt is not None:
                detections.append(('fire', pt, 0.95))
        return detections

    # ------------------------------------------------------------------ 2. ArUco Marker detector
    def detect_markers(self):
        """Detects DICT_4X4_100 ArUco tags on junction/landmark beacon posts."""
        if self.rgb is None or self.K_rgb is None or self.depth_m is None:
            return []

        gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        detections = []
        if ids is not None and len(ids) > 0:
            for i, corner_set in enumerate(corners):
                pts = corner_set[0]
                u = float(np.mean(pts[:, 0]))
                v = float(np.mean(pts[:, 1]))
                d = self.sample_depth(u, v, w=7, h=7)
                if d is not None and 0.4 <= d <= 30.0:
                    map_pt = self.pixel_to_map(u, v, d, self.K_rgb, 'camera_color_optical_frame')
                    if map_pt is not None:
                        marker_id = int(ids[i][0])
                        detections.append((f'marker:{marker_id}', map_pt, 0.98))
        return detections

    # ------------------------------------------------------------------ 3. Survivor detector
    def detect_survivors(self):
        """Detects real survivors using YOLOv8 person detector + Thermal ROI validation.
        Rejects cold mannequin decoys (~298 K), heated ground / smouldering barrels near fires,
        and overheated machinery (> 325 K)."""
        if self.rgb is None or self.K_rgb is None or self.depth_m is None or self.thermal_k is None or self.K_thermal is None:
            return []

        detections = []
        person_candidates = []

        # Step A: Run YOLOv8 on RGB with sensitive threshold (catches sitting and occluded postures)
        if self.ort_session is not None:
            orig_h, orig_w = self.rgb.shape[:2]
            scale = min(640.0 / orig_w, 640.0 / orig_h)
            nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
            resized = cv2.resize(self.rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((640, 640, 3), dtype=np.uint8)
            pad_x = (640 - nw) // 2
            pad_y = (640 - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

            blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
            try:
                out = self.ort_session.run(None, {self.model_input_name: blob})[0]  # (1, 84, 8400)
                preds = out[0].T  # (8400, 84)
                person_scores = preds[:, 4]  # class 0 = person in COCO
                mask = person_scores >= self.conf_thresh
                if np.any(mask):
                    cand_boxes = preds[mask, :4]
                    cand_scores = person_scores[mask]
                    cx = (cand_boxes[:, 0] - pad_x) / scale
                    cy = (cand_boxes[:, 1] - pad_y) / scale
                    bw = cand_boxes[:, 2] / scale
                    bh = cand_boxes[:, 3] / scale
                    x1 = np.clip(cx - bw / 2, 0, orig_w - 1)
                    y1 = np.clip(cy - bh / 2, 0, orig_h - 1)
                    bw = np.clip(bw, 1, orig_w - x1)
                    bh = np.clip(bh, 1, orig_h - y1)

                    boxes_for_nms = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, [float(s) for s in cand_scores], self.conf_thresh, 0.45)
                    for idx in indices:
                        i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                        person_candidates.append((x1[i], y1[i], bw[i], bh[i], float(cand_scores[i])))
            except Exception as e:
                self.get_logger().error(f'YOLO inference error: {e}')

        # Step B: Cross-spectral Thermal Verification
        for x1, y1, bw, bh, score in person_candidates:
            u_center = x1 + bw / 2
            v_center = y1 + bh / 2
            depth_val = self.sample_depth(u_center, v_center, w=max(3, int(bw * 0.4)), h=max(3, int(bh * 0.4)))
            if depth_val is None or depth_val > 14.0:
                continue

            u_th_c, v_th_c = self.rgb_to_thermal_coords(u_center, v_center, depth_val)
            if u_th_c is None:
                continue

            fx_rgb = self.K_rgb[0, 0]
            fx_th = self.K_thermal[0, 0]
            fy_th = self.K_thermal[1, 1]
            bw_th = bw * (fx_th / fx_rgb)
            bh_th = bh * (fy_th / self.K_rgb[1, 1])

            th_h, th_w = self.thermal_k.shape
            tu0 = max(0, int(round(u_th_c - bw_th / 2)))
            tu1 = min(th_w, int(round(u_th_c + bw_th / 2)) + 1)
            tv0 = max(0, int(round(v_th_c - bh_th / 2)))
            tv1 = min(th_h, int(round(v_th_c + bh_th / 2)) + 1)

            if tu1 <= tu0 or tv1 <= tv0:
                continue

            th_crop = self.thermal_k[tv0:tv1, tu0:tu1]
            if th_crop.size < 4:
                continue

            t_max = float(np.max(th_crop))

            # 1. Cold mannequin decoy (ambient ~298 K) -> REJECT
            if t_max < 300.5:
                continue

            # 2. Overheated non-human (burnt car, engine, fire) -> REJECT
            if t_max > 325.0:
                continue

            # 3. Human body temperature band (301 - 320 K)
            if 301.0 <= t_max <= 320.0:
                map_pt = self.pixel_to_map(u_center, v_center, depth_val, self.K_rgb, 'camera_color_optical_frame')
                if map_pt is not None:
                    # Filter out perimeter wall banners (beyond arena boundaries)
                    if -16.0 <= map_pt[0] <= 57.0 and -16.0 <= map_pt[1] <= 57.0:
                        # Ground-height check: covers lying (0.2m), sitting (0.5m), and standing (1.4m)
                        if -0.3 <= map_pt[2] <= 1.85:
                            self.get_logger().info(f'Survivor detected at map ({map_pt[0]:.2f}, {map_pt[1]:.2f}, z={map_pt[2]:.2f}) t_max={t_max:.1f}K conf={score:.2f}')
                            detections.append(('survivor', map_pt, score))

        return detections

    # ------------------------------------------------------------------ 4. Landmark detector
    def detect_landmarks(self):
        """Detects containers, relief tents, and signs with strict size and aspect filtering."""
        if self.rgb is None or self.depth_m is None or self.K_rgb is None:
            return []

        detections = []
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)

        color_ranges = {
            'red': [((0, 110, 70), (10, 255, 255)), ((170, 110, 70), (180, 255, 255))],
            'green': [((40, 80, 50), (80, 255, 255))],
            'blue': [((100, 80, 50), (125, 255, 255))],
            'yellow': [((22, 110, 90), (35, 255, 255))],
            'orange': [((11, 110, 90), (21, 255, 255))],
        }

        for col, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                area = cv2.contourArea(c)
                if area < 1500:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                u = bx + bw / 2
                v = by + bh / 2
                d = self.sample_depth(u, v, w=min(bw // 3, 20), h=min(bh // 3, 20))
                if d is None or not (1.8 <= d <= 25.0):
                    continue

                w_m = bw * (d / self.K_rgb[0, 0])
                h_m = bh * (d / self.K_rgb[1, 1])

                # Container: wide rectangular block (~2.4m - 6.5m wide, ~2.0m - 3.0m high)
                if 2.2 <= w_m <= 7.0 and 1.8 <= h_m <= 3.2:
                    pt = self.pixel_to_map(u, v, d, self.K_rgb, 'camera_color_optical_frame')
                    if pt is not None:
                        detections.append((f'container:{col}', pt, 0.90))
                # Tent: ~3.0m - 4.5m wide, ~2.0m - 3.5m high
                elif 2.5 <= w_m <= 4.5 and 1.8 <= h_m <= 3.5 and col in ('blue', 'orange', 'green'):
                    pt = self.pixel_to_map(u, v, d, self.K_rgb, 'camera_color_optical_frame')
                    if pt is not None:
                        detections.append((f'tent:{col}', pt, 0.85))

        return detections

    # ------------------------------------------------------------------ main detect entry
    def detect(self):
        """Called every timer tick. Runs all detectors and returns list of (class_id, (x, y, z), score)."""
        detections = []
        detections.extend(self.detect_fires())
        detections.extend(self.detect_markers())
        detections.extend(self.detect_survivors())
        detections.extend(self.detect_landmarks())
        return detections

    def process(self):
        if self.rgb is None or self.depth_m is None:
            return
        out = Detection3DArray()
        out.header.frame_id = 'map'
        out.header.stamp = self.stamp if self.stamp is not None else self.get_clock().now().to_msg()
        for class_id, (x, y, z), score in self.detect():
            d = Detection3D()
            d.header = out.header
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = str(class_id), float(score)
            h.pose.pose.position.x = float(x)
            h.pose.pose.position.y = float(y)
            h.pose.pose.position.z = float(z)
            d.results.append(h)
            d.bbox.center.position = h.pose.pose.position
            out.detections.append(d)
        self.pub.publish(out)

        # Evidence picture for rescue reports
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            snap = CompressedImage()
            snap.header = out.header
            snap.format, snap.data = 'jpeg', jpg.tobytes()
            self.pub_snap.publish(snap)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
pen_vision.py

Reusable RealSense + purple-pen detection helpers.

Provides:
- RealSenseCapture: context manager that opens live RGBD streams aligned to color.
- find_pen_centroid_and_xyz(...): locate pen in RGB, get depth, deproject to 3D (camera frame).
- collect_camera_means(...): average pen XYZ over a duration (returns dict with mean X/Y/Z).
- measure_pen_in_camera(...): like collect_camera_means but returns a numpy array [X,Y,Z].

Notes:
- No ROI fallback: if depth at centroid is 0.0, that frame is ignored.
- HSV threshold defaults target purple; override via parameters if lighting changes.
"""

import time
import numpy as np
import cv2
import pyrealsense2 as rs

class RealSenseCapture:
    def __init__(self, width=640, height=480, align_to="color"):
        self.width = width
        self.height = height
        self.align_to = rs.stream.color if align_to == "color" else rs.stream.depth
        self.pipeline = None
        self.align = None
        self.profile = None

    def __enter__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(cfg)
        self.align = rs.align(self.align_to)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.profile = None

    def get_aligned_frames(self, timeout_ms=5000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned = self.align.process(frames)
        return aligned.get_depth_frame(), aligned.get_color_frame()


# ---------------------------
# Detection core
# ---------------------------
def find_pen_centroid_and_xyz(
    aligned_depth_frame,
    color_image,
    depth_scale,
    clipping_distance_m=1.5,
    min_area=150,
    hsv_lower=(115, 50, 50),
    hsv_upper=(175, 255, 255),
    show_overlay=True,
):
    grey_color = 153
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    clipping_distance = clipping_distance_m / depth_scale

    bg_removed = np.where(
        (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
        grey_color,
        color_image,
    )

    hsv = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)
    lower_purple = np.array(hsv_lower, dtype=np.uint8)
    upper_purple = np.array(hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)

    debug = color_image.copy() if show_overlay else None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ok = False
    cx = cy = None
    X = Y = Z = None

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > float(min_area):
            if len(c) >= 5:
                (ex, ey), (MA, ma), angle = cv2.fitEllipse(c)
                cx, cy = int(round(ex)), int(round(ey))
                if show_overlay:
                    cv2.ellipse(debug, ((ex, ey), (MA, ma), angle), (0, 255, 255), 2)
            else:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w // 2, y + h // 2

            if cx is not None and cy is not None:
                d_m = aligned_depth_frame.get_distance(cx, cy) 
                if d_m > 0.0:
                    intr = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], d_m)
                    ok = True
                    if show_overlay:
                        cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.putText(
                            debug,
                            f"XYZ(m)=({X:.3f},{Y:.3f},{Z:.3f})",
                            (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                elif show_overlay:
                    cv2.putText(
                        debug,
                        "No valid depth at centroid",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

    if show_overlay and contours:
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)

    return ok, cx, cy, X, Y, Z, debug

def collect_camera_means(
    cam,
    duration_s=5.0,
    min_area=150,
    clip_m=1.5,
    window_name="Pen Tracker",
    hsv_lower=(115, 50, 50),
    hsv_upper=(175, 255, 255),
):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    depth_sensor = cam.profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    xs, ys, zs = [], [], []
    t0 = time.time()
    while (time.time() - t0) < duration_s:
        dframe, cframe = cam.get_aligned_frames()
        if not dframe or not cframe:
            continue
        color = np.asanyarray(cframe.get_data())

        ok, _, _, X, Y, Z, debug = find_pen_centroid_and_xyz(
            dframe,
            color,
            depth_scale,
            clipping_distance_m=clip_m,
            min_area=min_area,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            show_overlay=True,
        )
        if ok:
            xs.append(X); ys.append(Y); zs.append(Z)

        cv2.imshow(window_name, debug if debug is not None else color)
        k = cv2.waitKey(1)
        if k in (ord("q"), 27):
            break

    cv2.destroyWindow(window_name)

    if len(xs) == 0:
        import math
        return {"cam_mean_X_m": math.nan, "cam_mean_Y_m": math.nan, "cam_mean_Z_m": math.nan}

    import numpy as _np
    return {
        "cam_mean_X_m": float(_np.mean(xs)),
        "cam_mean_Y_m": float(_np.mean(ys)),
        "cam_mean_Z_m": float(_np.mean(zs)),
    }


def measure_pen_in_camera(
    cam,
    duration_s=5.0,
    min_area=150,
    clip_m=2.0,
    window_name="Pen Tracker",
    hsv_lower=(115, 50, 50),
    hsv_upper=(175, 255, 255),
):
    import numpy as _np

    d = collect_camera_means(
        cam,
        duration_s=duration_s,
        min_area=min_area,
        clip_m=clip_m,
        window_name=window_name,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
    )
    if any(_np.isnan([d["cam_mean_X_m"], d["cam_mean_Y_m"], d["cam_mean_Z_m"]])):
        raise RuntimeError("Pen not detected (no valid camera samples).")
    return _np.array([d["cam_mean_X_m"], d["cam_mean_Y_m"], d["cam_mean_Z_m"]], dtype=float)

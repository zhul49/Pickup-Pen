#!/usr/bin/env python3
import csv
import time
import math
import numpy as np
import cv2
import pyrealsense2 as rs

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_startup
from scipy.spatial.transform import Rotation as SciRot
from pen_vision import RealSenseCapture, measure_pen_in_camera


# ======== Hard-coded settings ========
OUT_CSV     = "pose_camera_summary.csv"
SETTLE_SECS = 0.75
HOLD_SECS   = 5.0
MIN_AREA    = 150        # px; increase if flicker
CLIP_M      = 1.5        # m; background cutoff for depth
POSES = [
    ("pos1", dict(x=0.19, y= 0.00, z=0.15, roll=0.0, pitch=0.8, yaw=0.0)),
    ("pos2", dict(x=0.22, y= 0.08, z=0.14, roll=0.0, pitch=0.8, yaw=0.0)),
    ("pos3", dict(x=0.22, y=-0.05, z=0.14, roll=0.0, pitch=0.8, yaw=0.0)),
    ("pos4", dict(x=0.12, y=-0.03, z=0.12, roll=0.0, pitch=0.8, yaw=0.0)),
    ("pos5", dict(x=0.11, y= 0.00, z=0.20, roll=0.0, pitch=0.8, yaw=0.0)),
]
WINDOW_NAME = "Pen Tracker"
# =====================================

def get_xyz(T):
    M = np.array(T, dtype=float)
    x, y, z = M[:3, 3]
    return float(x), float(y), float(z)

def find_pen_centroid_and_xyz(aligned_depth_frame, color_image, depth_scale,
                              clipping_distance_m=CLIP_M, min_area=MIN_AREA, show_overlay=True):
    """
    Returns (ok, cx, cy, X, Y, Z, debug_image)
    NOTE: No ROI fallback; if depth at centroid == 0.0, sample is discarded.
    """
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
    lower_purple = np.array([115, 50, 50], dtype=np.uint8)
    upper_purple = np.array([175, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)

    debug = color_image.copy() if show_overlay else None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = cy = None
    X = Y = Z = None
    ok = False

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > float(min_area):
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
                d_m = aligned_depth_frame.get_distance(cx, cy)  # meters
                if d_m > 0.0:
                    intr = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], d_m)
                    ok = True
                    if show_overlay:
                        cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.putText(debug, f"XYZ(m)=({X:.3f},{Y:.3f},{Z:.3f})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)
                else:
                    if show_overlay:
                        cv2.putText(debug, "No valid depth at centroid", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if show_overlay and contours:
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)

    return ok, cx, cy, X, Y, Z, debug

def collect_camera_means(cam, duration_s=HOLD_SECS, min_area=MIN_AREA, clip_m=CLIP_M):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    depth_sensor = cam.profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    xs, ys, zs = [], [], []
    t0 = time.time()

    while (time.time() - t0) < duration_s:
        try:
            aligned_depth_frame, color_frame = cam.get_aligned_frames()
        except rs.error as e:
            print(f"[RealSense] {e}")
            break

        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        ok, _, _, X, Y, Z, debug = find_pen_centroid_and_xyz(
            aligned_depth_frame,
            color_image,
            depth_scale,
            clipping_distance_m=clip_m,
            min_area=min_area,
            show_overlay=True,
        )

        if ok:
            xs.append(X); ys.append(Y); zs.append(Z)

        cv2.imshow(WINDOW_NAME, debug if debug is not None else color_image)
        k = cv2.waitKey(1)
        if k in (ord('q'), 27): 
            break

    cv2.destroyWindow(WINDOW_NAME)

    if len(xs) == 0:
        return {"cam_mean_X_m": math.nan, "cam_mean_Y_m": math.nan, "cam_mean_Z_m": math.nan}

    return {
        "cam_mean_X_m": float(np.mean(xs)),
        "cam_mean_Y_m": float(np.mean(ys)),
        "cam_mean_Z_m": float(np.mean(zs)),
    }

def solve_r_t_from_csv(csv_path: str):
    cam_pts, rob_pts = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rx = float(row["robot_x_m"]); ry = float(row["robot_y_m"]); rz = float(row["robot_z_m"])
                cx = float(row["cam_mean_X_m"]); cy = float(row["cam_mean_Y_m"]); cz = float(row["cam_mean_Z_m"])
            except (KeyError, ValueError):
                continue
            if any(np.isnan([rx, ry, rz, cx, cy, cz])):
                continue
            rob_pts.append([rx, ry, rz])
            cam_pts.append([cx, cy, cz])

    P = np.asarray(cam_pts, dtype=float)  
    Q = np.asarray(rob_pts, dtype=float) 
    if P.shape[0] < 3:
        raise SystemExit(f"Need at least 3 valid point pairs; found {P.shape[0]}")

    P_bar = P.mean(axis=0)
    Q_bar = Q.mean(axis=0)
    Pc = P - P_bar
    Qc = Q - Q_bar

    rot, _ = SciRot.align_vectors(Qc, Pc)
    R = rot.as_matrix()

    t = Q_bar - R @ P_bar
    return R, t

def append_rt_to_csv(csv_path: str, R: np.ndarray, t: np.ndarray):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([])
        w.writerow(["R00","R01","R02","R10","R11","R12","R20","R21","R22","T0","T1","T2"])
        row = [R[0,0],R[0,1],R[0,2], R[1,0],R[1,1],R[1,2], R[2,0],R[2,1],R[2,2], t[0],t[1],t[2]]
        w.writerow([f"{v:.9f}" for v in row])

def main():
    robot = InterbotixManipulatorXS("px100", "arm", "gripper")
    robot_startup()

    rows = []
    try:
        with RealSenseCapture(align_to="color") as cam:
            print("\n=== Starting pose capture with RealSense sampling ===")
            for i, (name, comp) in enumerate(POSES, start=1):
                print(f"\n=== Moving to {name} ===")
                robot.arm.set_ee_pose_components(**comp)
                time.sleep(SETTLE_SECS)

                T = robot.arm.get_ee_pose()
                rob_x, rob_y, rob_z = get_xyz(T)
                print(f"{name} robot EE (x,y,z) = ({rob_x:.4f}, {rob_y:.4f}, {rob_z:.4f})")

                cam_means = collect_camera_means(cam, duration_s=HOLD_SECS, min_area=MIN_AREA, clip_m=CLIP_M)
                print(f"{name} camera mean XYZ (m) = "
                      f"({cam_means['cam_mean_X_m']:.4f}, {cam_means['cam_mean_Y_m']:.4f}, {cam_means['cam_mean_Z_m']:.4f})")

                rows.append({
                    "pose_idx": i,
                    "robot_x_m": rob_x,
                    "robot_y_m": rob_y,
                    "robot_z_m": rob_z,
                    "cam_mean_X_m": cam_means["cam_mean_X_m"],
                    "cam_mean_Y_m": cam_means["cam_mean_Y_m"],
                    "cam_mean_Z_m": cam_means["cam_mean_Z_m"],
                })
    finally:
        try:
            robot.arm.go_to_sleep_pose()
        except Exception:
            robot.arm.go_to_sleep_pose()
            pass
        try:
            robot.shutdown()
        except Exception:
            robot.arm.go_to_sleep_pose()
            pass

    fieldnames = [
        "pose_idx",
        "robot_x_m", "robot_y_m", "robot_z_m",
        "cam_mean_X_m", "cam_mean_Y_m", "cam_mean_Z_m",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved summary to: {OUT_CSV}")

    R, t = solve_r_t_from_csv(OUT_CSV)
    append_rt_to_csv(OUT_CSV, R, t)
    print(f"Appended R and T to {OUT_CSV}")

if __name__ == "__main__":
    main()

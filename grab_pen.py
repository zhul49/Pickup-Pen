#!/usr/bin/env python3
import csv
import time
import math
import numpy as np
import cv2
import pyrealsense2 as rs

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import robot_startup
from pen_vision import RealSenseCapture, measure_pen_in_camera

CALIB_CSV = "pose_camera_summary.csv"

HOVER_DZ     = 0.055
TOOL_OFFSET_LOCAL = (0.05, 0.000, 0.070)

MIN_AREA  = 180
CLIP_M    = 1.5
HOLD_SECS = 2.0
WINDOW    = "Pen Tracker"

def load_rt_from_csv(csv_path: str):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    hdr_idx = None
    for i, row in enumerate(rows):
        if len(row) >= 12 and row[0] == "R00" and row[11] == "T2":
            hdr_idx = i
            break
    if hdr_idx is None or hdr_idx + 1 >= len(rows):
        raise SystemExit(f"Could not find R/T lines in {csv_path}.")
    vals = [float(x) for x in rows[hdr_idx + 1][:12]]
    R = np.array(vals[:9], dtype=float).reshape(3, 3)
    t = np.array(vals[9:12], dtype=float)
    return R, t

def cam_to_robot(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R @ P_cam + t

def ee_pose(robot, x, y, z):
    robot.arm.set_ee_pose_components(x=float(x), y=float(y), z=float(z))
    time.sleep(0.4)

def open_gripper(robot):
    try:
        robot.gripper.open()
    except AttributeError:
        robot.gripper.release()
    time.sleep(0.3)

def close_gripper(robot):
    try:
        robot.gripper.close()
    except AttributeError:
        robot.gripper.grasp()
    time.sleep(0.3)

def main():
    R, t = load_rt_from_csv(CALIB_CSV)
    robot = InterbotixManipulatorXS("px100", "arm", "gripper")
    robot_startup()
    try:
        with RealSenseCapture(align_to="color") as cam:
            while True:
                try:
                    P_cam = measure_pen_in_camera(
                        cam,
                        duration_s=HOLD_SECS,
                        min_area=MIN_AREA,
                        clip_m=CLIP_M,
                        window_name=WINDOW,
                    )
                except KeyboardInterrupt:
                    break
                except Exception:
                    continue
                if P_cam is None or len(P_cam) != 3:
                    continue

                P_rob = cam_to_robot(P_cam, R, t) + np.array(TOOL_OFFSET_LOCAL, dtype=float)
                rx, ry, rz = map(float, P_rob)

                open_gripper(robot)
                ee_pose(robot, rx, ry, rz + HOVER_DZ)
                ee_pose(robot, rx, ry, rz)
                close_gripper(robot)
                ee_pose(robot, rx, ry, rz + 0.03)
                open_gripper(robot)
                time.sleep(2.5)
    finally:
        try:
            robot.arm.go_to_sleep_pose()
        except Exception:
            pass
        try:
            robot.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()

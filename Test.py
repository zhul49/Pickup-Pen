#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs

class RealSenseCapture:
    def __init__(self, playback=None, record=None, width=640, height=480, align_to="color"):
        self.playback_file = playback
        self.record_file = record
        self.width = width
        self.height = height
        self.align_to = rs.stream.color if align_to == "color" else rs.stream.depth
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self._playback_dev = None

    def __enter__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.playback_file:
            try:
                self.config.enable_device_from_file(self.playback_file, False)
            except TypeError:
                self.config.enable_device_from_file(self.playback_file)
        else:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        if self.record_file and not self.playback_file:
            self.config.enable_record_to_file(self.record_file)

        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(self.align_to)

        if self.playback_file:
            dev = self.profile.get_device()
            self._playback_dev = dev.as_playback()
            self._playback_dev.set_real_time(False)

        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self._playback_dev = None

    def get_aligned_frames(self, timeout_ms=5000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        aligned_frames = self.align.process(frames)
        aligned_deapth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return aligned_deapth_frame, color_frame

    def get_images(self):
        aligned_deapth_frame, color_frame = self.get_aligned_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_deapth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return color_image, depth_image, depth_colormap

def main():
    p = argparse.ArgumentParser(description="RealSense: live stream or playback with aligned RGB/Depth")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--playback", type=str, help="Path to .bag file to play")
    mode.add_argument("--live", action="store_true", help="Use live camera (default if neither is set)")
    p.add_argument("--record", type=str, help="Record live stream to .bag (only valid with --live)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--align", choices=["color", "depth"], default="color")
    p.add_argument("--window", default="Aligned RGB | Depth", help="Window name")
    p.add_argument("--poll", action="store_true", help="Use cv2.pollKey() instead of waitKey(1) if available")
    p.add_argument("--quit-key", default="q", help="Key to quit (default: q)")
    args = p.parse_args()

    if args.playback and args.record:
        raise SystemExit("Cannot use --record while in --playback mode.")

    use_poll = args.poll and hasattr(cv2, "pollKey")
    quit_code = ord(args.quit_key) if len(args.quit_key) == 1 else 27

    with RealSenseCapture(
        playback=args.playback,
        record=args.record,
        width=args.width,
        height=args.height,
        align_to=args.align,
    ) as cam:
        cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

        depth_sensor = cam.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1.5
        clipping_distance = clipping_distance_in_meters / depth_scale
        grey_color = 153

        smoothed_px, smoothed_py = None, None
        ema_keep = 0.8 

        while True:
            try:
                aligned_deapth_frame, color_frame = cam.get_aligned_frames()
                aligned_depth_frame = aligned_deapth_frame
            except rs.error as e:
                print(f"[RealSense] {e}")
                break

            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
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

            purple_filter = cv2.bitwise_and(bg_removed, bg_removed, mask=mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(purple_filter, contours, -1, (0, 255, 0), 1)

            if contours:
                # largest contour only
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > 150:
                    cx, cy = None, None

                    if len(c) >= 5:
                        (ex, ey), (MA, ma), angle = cv2.fitEllipse(c)
                        cx, cy = int(round(ex)), int(round(ey))
                        cv2.ellipse(purple_filter, ((ex, ey), (MA, ma), angle), (0, 255, 255), 2)
                    else:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            x, y, w, h = cv2.boundingRect(c)
                            cx, cy = x + w // 2, y + h // 2


                    d = aligned_depth_frame.get_distance(cx, cy)
                    if d == 0.0:
                        roi = depth_image[max(cy - 3, 0): cy + 4, max(cx - 3, 0): cx + 4]
                        nz = roi[roi > 0]
                        if nz.size > 0:
                            d = float(np.median(nz) * depth_scale)

                    if d > 0.0:
                        intr = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], d)
                        cv2.circle(purple_filter, (cx, cy), 6, (0, 0, 255), -1)
                        overlay = f"px=({cx},{cy})  XYZ(m)=({X:.3f}, {Y:.3f}, {Z:.3f})"
                        cv2.putText(purple_filter, overlay, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        print(f"Centroid px=({cx},{cy})  XYZ [m]=({X:.4f}, {Y:.4f}, {Z:.4f})")
                    else:
                        cv2.putText(purple_filter, "No valid depth at centroid", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            images = np.hstack((purple_filter, color_image))
            cv2.imshow(args.window, images)

            k = cv2.pollKey() if use_poll else cv2.waitKey(1)
            if k == quit_code or k == 27:
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()







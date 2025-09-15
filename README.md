# Pen Pick-Up Test

This project demonstrates how to calibrate an Interbotix PX100 robot arm with an Intel RealSense camera and perform a pen pick-up using vision tracking.

---

## Prerequisites

- ROS 2 installed and configured  
- Interbotix PX100 arm  
- Intel RealSense camera  
- Interbotix ROS packages built in your workspace (`ws`)  

---

## How to Run

### 1. Launch the Robot Control Node
In a **separate terminal**, start the Interbotix control node:

`ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px100`

This launches the ROS 2 driver for the PX100 arm.
2. Source the Workspace

In another terminal, source your Interbotix workspace:

`source ws/interbotix/install/setup.bash`

This ensures that all required ROS 2 packages and dependencies are available.
3. Calibrate the Camera and Robot

Run the calibration script:

`python3 main.py`

The robot moves to several pre-set positions.

The RealSense camera observes the pen.

Calibration parameters are computed to transform coordinates between the camera and robot frames.

4. Track and Grab the Pen

Once calibration is complete, run:

`python3 grab_pen.py`

The RealSense camera detects and tracks the pen in real time.

The robot uses the calibration to move its end effector toward the pen and attempt to grab it.

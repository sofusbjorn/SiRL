from optimizer import *

import mujoco
import mujoco.viewer
import time
import rtde_receive
from pathlib import Path
import numpy as np
import rtde_control 




ROBOT_IP = "192.168.1.103"
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

joint_q = rtde_r.getActualQ()
print("Joint positions [rad]:", joint_q)

# Load model from XML files
_here = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(_here / 'scene.xml'))
data = mujoco.MjData(model)
data.qpos[:] = joint_q
mujoco.mj_forward(model, data)
print(f'Initial joint positions: {data.qpos}')


left = 263
right = 262
up = 265
down = 264
comma = 44
period = 46

target_q = data.qpos.copy()
print(f'Initial target joint positions: {target_q}')


def key_callback(keycode):
    pos, mat = forward_kinematics(data.qpos)
    global target_q
    newPos = pos.copy()

    print(data.qpos)
    if keycode == left:  # Left arrow key
        print("Left arrow key pressed!")
        newPos = pos + np.array([-0.005, 0, 0])
    if keycode == right:  # Right arrow key
        print("Right arrow key pressed!")
        newPos = pos + np.array([0.005, 0, 0])
    if keycode == up:  # Up arrow key
        print("Up arrow key pressed!")
        newPos = pos + np.array([0, 0, 0.005])
    if keycode == down:  # Down arrow key
        print("Down arrow key pressed!")
        newPos = pos + np.array([0, 0, -0.005])
    if keycode == comma:  # Comma key
        print("Comma key pressed!")
        newPos = pos + np.array([0, -0.005, 0])
    if keycode == period:  # Period key
        print("Period key pressed!")
        newPos = pos + np.array([0, 0.005, 0])

    target_q = ik_optim(newPos, mat, data.qpos)

# Launch a viewer
try:
    # Launch a viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        pos, mat = forward_kinematics(data.qpos)
        target_q = ik_optim(pos, mat, data.qpos)
        while viewer.is_running():
            
            data.qpos[:] = target_q
            
            # 2. Update visual/spatial positions
            mujoco.mj_forward(model, data) 

            # 3. Send the new joint positions to the robot
            # Change dt to 0.01 to match your sleep timer
            rtde_c.servoJ(target_q, 0.0, 0.0, 0.01, 0.1, 300)
            
            # 4. Sync the viewer
            viewer.sync()
            
            # 5. Sleep for 10ms instead of 1000ms
            time.sleep(0.01)

finally:
    # Cleanup - always runs even if exception occurs
    if rtde_c.isConnected():
        rtde_c.disconnect()
    if rtde_r.isConnected():
        rtde_r.disconnect()

#disconnect from the robot when control loop is exited or the viewer is closed or when control c is pressed











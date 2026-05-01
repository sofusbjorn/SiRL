import mujoco
import mujoco.viewer
import time
import rtde_receive
from pathlib import Path

# Load model from XML files
_here = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(_here / 'scene.xml'))
data = mujoco.MjData(model)

# Connect to the robot using RTDE
ROBOT_IP = "192.168.1.102"
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)


def update_model_from_robot(model, data, rtde_r):
    # Get the current joint positions from the robot
    joint_positions = rtde_r.getActualQ()
    
    # Update the model's joint positions
    data.qpos = joint_positions
    
    return data



# Launch a viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Update the model's joint positions from the robot
        data = update_model_from_robot(model, data, rtde_r)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


# Disconnect from the robot
rtde_r.disconnect()

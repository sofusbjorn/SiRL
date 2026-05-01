import mujoco
import mujoco.viewer
import time
import rtde_receive 

# Load model from XML files
model = mujoco.MjModel.from_xml_path("scene.xml")  # or model.xml
data = mujoco.MjData(model)

# Connect to the robot using RTDE
ROBOT_IP = "192.168.1.102"
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)


def update_model_from_robot(model, data, rtde_r):
    # Get the current joint positions from the robot
    joint_positions = rtde_r.getActualQ()
    print("Current joint positions from robot: ", joint_positions, flush=True)
    
    # Update the model's joint positions
    print("Current joint positions in model: ", data.qpos, flush=True)
    data.qpos = joint_positions
    
    return data



# Launch a viewer
with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        print("Updating model from robot...", flush=True)
        # Update the model's joint positions from the robot
        data = update_model_from_robot(model, data, rtde_r)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)


# Disconnect from the robot
rtde_r.disconnect()

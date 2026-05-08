import mujoco
from pathlib import Path
from spatial_algebra import Transform, Motion
import math_utils as math

# Load model from XML files
_here = Path(__file__).parent
model = mujoco.MjModel.from_xml_path(str(_here / 'threelinks.xml'))
data = mujoco.MjData(model)



# # Launch a viewer
# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running():
#         # Update the model's joint positions from the robot
#         mujoco.mj_step(model, data)
#         print(model.geom_pos)
#         print(model.geom_quat)
#         viewer.sync()
#         time.sleep(0.01)


def transform_links(model, data):

    #uncomment to test the results when some joints are not at zero position
    #data.qpos[model.jnt_qposadr[1]] = 0.5

    trans_list= []
    # for each n body: 
    for i in range(1,model.nbody):
        #find angle the joint is rotated by
        angle = data.qpos[i-1]
        #Find which axes the joint is rotating around
        hinge_axis = model.jnt_axis[i-1]
        #Convert to quaternion
        rot = math.quat_rot_axis(hinge_axis, angle)
        #Find local transform. Since body_pos and data.qpos are relative to parents, this gives us 
        #The transform we were overthinking the entire day :)
        Ti = Transform(trans=model.body_pos[i], rot=rot)
        #append to list of transforms
        trans_list.append(Ti)
        
    return trans_list

def spatial_velocities(model, data):
    #Retrieve the transforms for each link
    trans_list = transform_links(model, data)
    #Instantiate Vi list - V0 is velocity of world, which is zero
    vel_list = [Motion()]

    for i in range(1, model.nbody):    
        #We assume all joints are hinge joints
        if model.jnt_type[i-1] == 3:
            #Retrieve the joint velocity and axis
            qi = data.qvel[i-1]
            Si = model.jnt_axis[i-1]
            #Calculate velocity applied to current link by its parent ---- V_{i, i-1} = V_{i-1} ^{i-1}T_i
            V_carried = vel_list[i-1].apply_transform(trans_list[i-1]) 
            #Add velocity contributed by the joint itself ---- V_i = V_{i, i-1} + S_i * q_i
            Vi = V_carried + Motion(ang=Si*qi)
            #Append to list of velocities
            vel_list.append(Vi)
        else: 
            raise NotImplementedError("Only hinge joints are supported in this function for now")

    return vel_list


if __name__ == "__main__":
    mujoco.mj_forward(model, data)

    print(transform_links(model, data))
    #print(spatial_velocities(model, data))


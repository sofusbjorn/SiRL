from articulated_system import ArticulatedSystem
from mujoco import MjModel, MjData
from pathlib import Path
from spatial_algebra import Transform, Motion
import math_utils as math
import numpy as np

# Load model from XML files
_here = Path(__file__).parent
model = MjModel.from_xml_path(str(_here / 'threelinks.xml'))
data = MjData(model)


def transform_links(sys: ArticulatedSystem, q: list[float]):
    trans_list= []
    # for each n body: 
    for i in range(sys.num_links()):
        link = sys.links[i]
        #find angle the joint is rotated by
        angle = q[i]
        #Find which axes the joint is rotating around
        hinge_axis = link.joint_dofs.motion.ang
        #Convert to quaternion
        rot = math.quat_rot_axis(hinge_axis, angle)
        #transform before rot
        ang_T = Transform(rot=rot)
        Ti = link.prev_transform.apply_transform(
            ang_T
        ).apply_transform(
            link.joint_frame.create_inverse()
        )
        #append to list of transforms
        trans_list.append(Ti)
        
    return trans_list

def spatial_velocities(
    sys: ArticulatedSystem,  trans_list: list[Transform], qd: list[float]
):
    #Instantiate Vi list - V0 is velocity of world, which is zero
    vel_list = [Motion()]

    for i in range(sys.num_links()):
        #We assume all joints are hinge joints
        #Retrieve the joint velocity and axis
        link = sys.links[i]
        qi = qd[i]
        Si = link.joint_dofs.motion.ang
        #Calculate velocity applied to current link by its parent ---- V_{i, i-1} = V_{i-1} ^{i-1}T_i
        V_carried = vel_list[i].apply_transform(trans_list[i]) 
        #Add velocity contributed by the joint itself ---- V_i = V_{i, i-1} + S_i * q_i
        Vi = V_carried + Motion(ang=Si*qi)
        #Append to list of velocities
        vel_list.append(Vi)

    return vel_list

def forward_kinematics(
    sys: ArticulatedSystem,
    q: list[float],
    qd: list[float],
) -> tuple[list[Transform], list[Motion]]:
    trans_list = transform_links(sys, q)
    vel_list = spatial_velocities(sys, trans_list, qd)

    return trans_list, vel_list


if __name__ == "__main__":
    from ThreeLinks import ThreeLinks

    print("Testing forward kinematics on ThreeLinks system:")
    print("-----------------------------------------------\n")
    sys = ThreeLinks()
    q = [0., 0., 1.]
    qd = [0., 1., 1.]
    trans_list, vel_list = forward_kinematics(sys, q, qd)
    for i, (T, V) in enumerate(zip(trans_list, vel_list[1:], strict=True)):
        print(f"Link {i}:")
        print(f"Transform:\n{T}")
        print(f"Velocity:\n{V}\n")

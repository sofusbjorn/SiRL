import numpy as np 
from week3.task2.read_from_mujoco import forward_kinematics

def forward_dynamics_aba(sys, q, qd, tau):
    #refer to featherstone's pseudo code in pp. 132 Rigid Body Dynamics
    #first loop - calculate link poses and velocity, note the result
    link_poses, link_vel = forward_kinematics()
    # velocity should be with the reference of the center-of-mass
    #put your process here
    # initialize articulated body inertia and bias terms related to the
    # handle body
    I_a_lst = [l.inertia for l in sys.links]
    #here if the forces applied to specific link need to be accounted, it
    # should be v \times Iv - f, note the reference frame of f
    p_a_lst = [xd_ss_lst[i].cross_force(I_a_lst[i].mul(xd_ss_lst[i])) for
               i in range(sys.num_links())]
    S_ss_lst = [sys.links[i].joint_dofs.motion.apply_inv_transform(
                sys.links[i].joint_frame) for i in range(sys.num_links())] #transform
    # screw axis to link frame
    zeta_lst = [xd_ss_lst[i].cross_motion(S_ss_lst[i]*qd[..., i]) for i
                in range(sys.num_links())] #assuming only 1-dof joint
    #second loop - backward to complete articulated body inertia and bias
    D_lst = [None] * sys.num_links()
    U_lst = [None] * sys.num_links()
    u_lst = [None] * sys.num_links()
    #put your process here. Note that the loop starts from the last link
    # and be careful about the reference frames of quantities involved
    # in computation!!
    #third loop - calculate joint acc
    a_lst = [None] * sys.num_links()
    qdd = []
    #put your process here. Remember to initialize acceleration for "zero
    # -th link"
    #which is a dummy one as the predecessor of the first physical link.
    return np.concatenate(qdd, axis=-1).squeeze()

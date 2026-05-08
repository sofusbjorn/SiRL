from typing import Any
import numpy as np 
from numpy.typing import NDArray
from read_from_mujoco import forward_kinematics
from articulated_system import ArticulatedSystem
from spatial_algebra import Inertia, Motion
from ThreeLinks import ThreeLinks

def forward_dynamics_aba(
    sys: ArticulatedSystem,
    q: list[float],
    qd: list[float],
    tau,
) -> NDArray[Any]:
    # Implements Featherstone's Articulated Body Algorithm (ABA).
    # See pseudocode on p.132 of "Rigid Body Dynamics Algorithms".
    # Given joint positions q, velocities qd, and applied torques tau,
    # this returns joint accelerations qdd.

    # --- Loop 1 (forward, root → tip): compute poses and velocities ---
    #
    # link_poses : list of spatial transforms, one per link.
    #   link_poses[i] is the 6×6 (or equivalent) transform that maps vectors
    #   from the world/base frame into link i's body frame.
    #
    # link_vel : list of spatial velocities (6-vectors [ω; v]), one per link,
    #   as returned by forward_kinematics. The frame these are expressed in
    #   depends on the forward_kinematics implementation — check the docstring.
    link_poses, link_vel = forward_kinematics(sys, q, qd)

    # link_vel : list of spatial velocities, one per link, expressed in each
    #   link's center-of-mass (CoM) body frame. This is what the ABA needs
    #   because the rigid-body inertia tensors (sys.links[i].inertia) are
    #   defined with respect to the CoM frame.
    #   If link_vel is already in the CoM frame you can use it directly;
    #   otherwise use link_poses[i] to transform link_vel[i] into the CoM frame.
    #
    # TODO: build link_vel (a list of length sys.num_links()) here.
    # Example structure (replace with correct transform):
    #   link_vel = [<transform link_vel[i] to CoM frame> for i in range(sys.num_links())]

    # I_a_lst : list of articulated-body inertia matrices, one per link.
    #   Initialized to each link's own rigid-body inertia (Featherstone eq. 7.2a).
    #   Loop 2 will augment these to include the contributions of all descendants.
    I_a_lst = [l.inertia for l in sys.links]

    # p_a_lst : list of articulated-body bias forces (6-vectors), one per link.
    #   p_a_i = v_i × (I_i * v_i)  (Featherstone eq. 7.2b)
    #   The cross_force(·) call computes the spatial cross-product v × f, which
    #   yields the gyroscopic/centrifugal force that resists the link's rotation.
    #   If an external wrench f_ext acts on link i, subtract it:
    #     p_a_i = v_i × (I_i * v_i) - f_ext_i
    #   f_ext must be in the same CoM frame as link_vel[i]. No external forces
    #   are present here, so the subtraction is skipped.
    p_a_lst = [link_vel[i].cross_force(I_a_lst[i].mul(link_vel[i])) for
               i in range(sys.num_links())]

    # S_ss_lst : list of joint motion subspaces (screw axes), one per link.
    #   S_i is a 6-vector describing which direction in spatial motion space
    #   joint i's single DOF acts (e.g. pure rotation around z → [0,0,1,0,0,0]).
    #   apply_inv_transform re-expresses S from the joint frame into the link's
    #   CoM body frame so it is consistent with I_a_lst and p_a_lst.
    S_ss_lst = [sys.links[i].joint_dofs.motion.apply_inv_transform(
                sys.links[i].joint_frame) for i in range(sys.num_links())]

    # zeta_lst : list of joint-bias accelerations (6-vectors), one per link.
    #   zeta_i = v_i × (S_i * qd_i)  (Featherstone eq. 7.36)
    #   This is the acceleration a link would experience even if qdd_i = 0,
    #   purely because the joint is moving (qd_i ≠ 0) while the link has
    #   velocity v_i. Conceptually it is the Coriolis/centrifugal term for
    #   the joint. It is added back in Loop 3 when computing link accelerations.
    #   Each joint is assumed to have exactly 1 DOF (hence qd[..., i]).
    zeta_lst = [link_vel[i].cross_motion(S_ss_lst[i]*qd[..., i]) for i
                in range(sys.num_links())]

    # --- Loop 2 (backward, tip → root): accumulate articulated-body inertia ---
    #
    # U_lst[i] = I_a_lst[i] * S_ss_lst[i]
    #   A 6-vector (force) that maps the joint axis into force space.
    #   Think of it as "how much force does a unit joint acceleration produce?"
    #
    # D_lst[i] = S_ss_lst[i]^T * U_lst[i]
    #   A scalar: the apparent inertia seen at joint i, accounting for all
    #   descendant links. Used to divide out joint acceleration later.
    #
    # u_lst[i] = tau[i] - S_ss_lst[i]^T * p_a_lst[i]
    #   A scalar: the net generalized force at joint i after subtracting the
    #   bias (gyroscopic + child-propagated) forces.
    #
    # After computing U, D, u for link i, propagate I_a and p_a upward to
    # link i's parent (Featherstone eqs. 7.25–7.26) using the spatial
    # transform X_{i→parent} that expresses link i's CoM frame in the parent's
    # CoM frame. Be careful: X must match the frame of I_a and p_a.
    #
    # Iterate from i = sys.num_links()-1 down to 0 (leaves first, root last).
    D_lst = [None] * sys.num_links()
    U_lst = [None] * sys.num_links()
    u_lst = [None] * sys.num_links()
    for i in range(sys.num_links() - 1, -1, -1):
        U_lst[i] = I_a_lst[i].mul(S_ss_lst[i])                        # Force: I_a * S
        D_lst[i] = S_ss_lst[i].dot_force(U_lst[i])                    # scalar: S^T * U
        u_lst[i] = tau[..., i] - S_ss_lst[i].dot_force(p_a_lst[i])   # scalar: tau - S^T * p_a

        parent = sys.parents[i]
        if parent >= 0:
            # Reduced articulated-body inertia: I_a_i - U_i * D_i^{-1} * U_i^T
            # (Featherstone eq. 7.25). from_dyad builds the outer-product term.
            I_a_reduced = I_a_lst[i] - Inertia.from_dyad(U_lst[i]) * (1.0 / D_lst[i])
            I_a_lst[parent] = I_a_lst[parent] + I_a_reduced.apply_transform(link_poses[i])

            # Propagate bias force to parent (Featherstone eq. 7.26).
            # apply_transform with the child→parent transform maps a force from
            # child frame to parent frame (dual/transpose of the motion transform).
            p_a_prop = (p_a_lst[i]
                        + I_a_lst[i].mul(zeta_lst[i])
                        + U_lst[i] * (u_lst[i] / D_lst[i]))
            p_a_lst[parent] = p_a_lst[parent] + p_a_prop.apply_transform(link_poses[i])

    # --- Loop 3 (forward, root → tip): compute joint accelerations ---
    #
    # a_lst[i] : spatial acceleration of link i in its CoM body frame.
    #
    # qdd      : list of scalar joint accelerations, one per link (1-DOF joints).
    #
    # Recurrence (Featherstone eqs. 7.27–7.28):
    #   qdd_i = (u_lst[i] - U_lst[i]^T * a_lst[parent_i]) / D_lst[i]
    #   a_lst[i] = X_{parent→i} * a_lst[parent_i] + S_ss_lst[i]*qdd_i + zeta_lst[i]
    #
    # The "zeroth" link is a virtual fixed base with no joint of its own.
    # Its spatial acceleration a_0 encodes gravity: set it to the negative of
    # the gravitational acceleration expressed as a spatial vector in the base
    # frame, e.g. a_0 = [0, 0, 0, 0, 0, -9.81] for z-up gravity.
    # (Using −g here means every link automatically "feels" gravity without
    # needing to add it as an external force separately.)
    # If gravity is not modelled, set a_0 = zero vector.
    a_lst = [None] * sys.num_links()
    qdd = []
    # Base acceleration: negative gravity as a spatial motion vector (Featherstone p.132).
    # Only the linear part is non-zero; angular part is zero for a fixed base.
    a_0 = Motion(lin=-sys.gravity)

    for i in range(sys.num_links()):
        parent = sys.parents[i]
        a_parent = a_lst[parent] if parent >= 0 else a_0

        # Bring parent acceleration into link i's CoM frame.
        a_parent_in_i = a_parent.apply_transform(link_poses[i])

        # Solve for joint acceleration (Featherstone eq. 7.28).
        # dot_force gives the scalar S^T * a_parent, i.e. the component of
        # parent acceleration along this joint's motion subspace.
        qdd_i = (u_lst[i] - a_parent_in_i.dot_force(U_lst[i])) / D_lst[i]

        # Link acceleration = parent-carried acceleration + joint contribution
        # + Coriolis bias (Featherstone eq. 7.27).
        a_lst[i] = a_parent_in_i + S_ss_lst[i] * qdd_i + zeta_lst[i]

        qdd.append(qdd_i[..., None])

    return np.concatenate(qdd, axis=-1).squeeze()


if __name__ == "__main__": 
    sys = ThreeLinks()
    g = [0,0,1] 
    gd=[0,0,0] 


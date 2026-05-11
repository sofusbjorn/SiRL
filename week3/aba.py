from typing import Any
import numpy as np 
from numpy.typing import NDArray
from forward_kin import forward_kinematics
from articulated_system import ArticulatedSystem
from spatial_algebra import Inertia, Motion
from ThreeLinks import ThreeLinks

def forward_dynamics_aba(
    sys: ArticulatedSystem,
    q: NDArray[Any],
    qd: NDArray[Any],
    tau: NDArray[Any],
    verbose: bool = False,
) -> NDArray[Any]:
    """
    Implements Featherstone's Articulated Body Algorithm (ABA).
    See pseudocode on p.132 of "Rigid Body Dynamics Algorithms".
    Given joint positions q, velocities qd, and applied torques tau,
    this returns joint accelerations qdd.
    """

    link_poses, full_link_vel = forward_kinematics(sys, q, qd)
    link_vel = full_link_vel[1:]

    I_a_lst = [l.inertia for l in sys.links]
    p_a_lst = [link_vel[i].cross_force(I_a_lst[i].mul(link_vel[i])) for
               i in range(sys.num_links())]
    S_ss_lst = [sys.links[i].joint_dofs.motion.apply_inv_transform(
                sys.links[i].joint_frame) for i in range(sys.num_links())]
    zeta_lst = [link_vel[i].cross_motion(S_ss_lst[i]*qd[i]) for i
                in range(sys.num_links())]

    # --- Loop 2 (backward, tip → root): accumulate articulated-body inertia ---
    D_lst = [None] * sys.num_links()
    U_lst = [None] * sys.num_links()
    u_lst = [None] * sys.num_links()
    for i in range(sys.num_links() - 1, -1, -1):
        U_lst[i] = I_a_lst[i].mul(S_ss_lst[i])                        # Force: I_a * S
        D_lst[i] = S_ss_lst[i].dot_force(U_lst[i])                    # scalar: S^T * U
        u_lst[i] = tau[i] - S_ss_lst[i].dot_force(p_a_lst[i])   # scalar: tau - S^T * p_a

        parent = sys.parents[i]
        if parent >= 0:
            I_a_reduced = I_a_lst[i] - Inertia.from_dyad(U_lst[i]) * (1.0 / D_lst[i])
            I_a_lst[parent] = I_a_lst[parent] + I_a_reduced.apply_transform(link_poses[i].create_inverse())

            p_a_prop = (p_a_lst[i]
                        + I_a_reduced.mul(zeta_lst[i])
                        + U_lst[i] * (u_lst[i] / D_lst[i]))
            p_a_lst[parent] = p_a_lst[parent] + p_a_prop.apply_transform(link_poses[i].create_inverse())

    # --- Loop 3 (forward, root → tip): compute joint accelerations ---
    a_lst = [None] * sys.num_links()
    qdd = []
    # Base acceleration: negative gravity as a spatial motion vector (Featherstone p.132).
    # Only the linear part is non-zero; angular part is zero for a fixed base.
    a_0 = Motion(lin=-sys.gravity)

    for i in range(sys.num_links()):
        parent = sys.parents[i]
        a_parent = a_lst[parent] if parent >= 0 else a_0

        # Bring parent acceleration into link i's CoM frame, then add the
        # Coriolis bias c_i before solving for q̈ (Featherstone Algorithm 7.4).
        a_tilde_i = a_parent.apply_transform(link_poses[i]) + zeta_lst[i]

        # Solve for joint acceleration (Featherstone eq. 7.28).
        qdd_i = (u_lst[i] - a_tilde_i.dot_force(U_lst[i])) / D_lst[i]

        a_lst[i] = a_tilde_i + S_ss_lst[i] * qdd_i

        qdd.append(qdd_i[..., None])

    if verbose:
        print("\n=== DEBUG (q=0, qd=0, tau=0) ===")
        print("S_ss_lst:")
        for i, S in enumerate(S_ss_lst):
            print(f"  link {i}: ang={S.ang}, lin={S.lin}")

        print("\nlink_vel (CoM frame):")
        for i, v in enumerate(link_vel):
            print(f"  link {i}: ang={v.ang}, lin={v.lin}")

        print("\nlink_poses (child→parent transforms):")
        for i, T in enumerate(link_poses):
            print(f"  link {i}: trans={T.trans}, rot={T.rot}")

        print("\nLoop 2 results:")
        for i in range(sys.num_links()):
            print(f"  link {i}: D={D_lst[i]:.4f}, u={u_lst[i]:.4f}")

        print("\na_0:", a_0)

    return np.concatenate(qdd, axis=-1).squeeze()


if __name__ == "__main__": 
    import mujoco
    from pathlib import Path

    # Load model from XML files
    _here = Path(__file__).parent
    model = mujoco.MjModel.from_xml_path(str(_here / 'threelinks.xml'))
    data = mujoco.MjData(model)

    sys = ThreeLinks()

    q = np.array([1,0.4,0.6] )
    qd = np.array([3,1,2])
    tau = np.array([1,2,0])
    
    qdd = forward_dynamics_aba(sys=sys, q=q, qd=qd, tau=tau)

    data.qpos = q
    data.qvel = qd
    data.qfrc_applied = tau

    mujoco.mj_forward(model, data)

    print("\n=== RESULTS ===")
    print(f"q: {q}, qd: {qd}, tau: {tau}")
    print("qdd (Ours):", qdd)
    print("qdd (MuJoCo):", data.qacc)

import scipy.optimize as optim
import jax
import mujoco
from mujoco import mjx
import jax.numpy as jnp
from pathlib import Path

ur5epath = str(Path(Path.cwd()/'week4'/'ur5e.xml'))

mj_model = mujoco.MjModel.from_xml_path(ur5epath)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)


def forward_kinematics(q):
    new_mjx_data= mjx_data.replace(qpos=q)
    new_mjx_data= mjx.fwd_position(mjx_model, new_mjx_data)
    pos = new_mjx_data.site_xpos[mj_model.site('attachment_site').id]
    mat = new_mjx_data.site_xmat[mj_model.site('attachment_site').id]
    return pos, mat

def pose_err(q, target_pos, target_mat):
    pos, mat = forward_kinematics(q)

    w_pos = 0.75
    w_rot = 0.25

    pos_err = jnp.sum((target_pos - pos) ** 2)
    rot_err = jnp.sum((target_mat - mat) ** 2)

    diff_scalar = w_pos * pos_err + w_rot * rot_err

    return diff_scalar


#jit compilation to make the run much faster
jit_err = jax.jit(pose_err)
#just get the gradient with auto-grad feature, with respect to the joint position guess (t
jit_err_grad = jax.jit(jax.grad(pose_err, argnums=(0)))

def ik_optim(target_pos, target_mat, init_guess):
    result = optim.minimize(jit_err, args=(target_pos, target_mat), x0=init_guess, method='trust-constr', jac=jit_err_grad)
    return result.x


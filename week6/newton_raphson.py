import jax
import jax.numpy as jnp


def newton_raphson(residual, init_guess, tol=1e-8, max_itr=100):
    """Find the root of a function using the Newton-Raphson method."""
    def cond_fn(state):
        res, _, _, i = state

        return (jnp.linalg.norm(res) > tol) & (i < max_itr)
    def body_fn(state):
        _, _, x, i = state
        res = residual(x)
        jac = jax.jacfwd(residual)(x)
        x_next = x - jnp.linalg.solve(jac, res)

        return (res, jac, x_next, i+1)
    state_init = (
        jnp.full_like(init_guess, 100),
        jnp.zeros(init_guess.shape[:-1] + (init_guess.shape[-1], init_guess.shape[-1])),
        init_guess,
        0
    )
    _, jac_x_star, x_star, _ = jax.lax.while_loop(cond_fn, body_fn, state_init)

    return x_star, jac_x_star

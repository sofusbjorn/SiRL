import jax
import jax.numpy as jnp
import jax.random as jr
import timeit

jax.devices("cpu")[0]

### WARMUP Of Assignment ###
def sigmoid_network(W, x):
    """Given weight matrix W and input x get sigmoid network y"""
    assert W.shape[1] == x.shape[0], (
        "Incompatible shapes for matrix multiplication"
    )
    return jax.nn.sigmoid(W @ x)

jacobian_f = jax.jacfwd(sigmoid_network, argnums=1) # w.r.t. x
jacobian_b = jax.jacrev(sigmoid_network, argnums=1) # w.r.t. x

### Warmup/Cache run
key = jr.PRNGKey(0)
m = 5000
n = 5000
W = jr.uniform(key, shape=(m, n), minval=0.0, maxval=1.0)
x = jr.uniform(key, shape=(n,), minval=0.0, maxval=1.0)


# Time jacobian_f
print("Warming up...")
jacobian_f(W, x)
jacobian_b(W, x)
###################

# Timing benchmarks
m = 5000
n = 5
configs = [(m, n), (n, m)]

print("Running test...")
for i, (m, n) in enumerate(configs):
    key = jr.PRNGKey(0)
    W = jr.uniform(key, shape=(m, n), minval=0.0, maxval=1.0)
    x = jr.uniform(key, shape=(n,), minval=0.0, maxval=1.0)

    print(f"--- m={m} and n={n} ---")

    # Time jacobian_f
    time_f = timeit.timeit(
        lambda: jacobian_f(W, x),
        number=10
    )
    print(f"jacobian_f: {time_f:.6f}s")

    # Time jacobian_b
    time_b = timeit.timeit(
        lambda: jacobian_b(W, x),
        number=10
    )
    print(f"jacobian_b: {time_b:.6f}s")

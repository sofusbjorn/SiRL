import jax
import jax.numpy as jnp
import timeit
from memory_profiler import profile

class rosenbrock_func:
    def __init__(self, a):
        self.a = a

    def rosenbrock(self, x):
        return (self.a[0] - x[0]) ** 2 + self.a[1] * (x[1] - x[0] ** 2) ** 2

    def gradient(self, x):
        dx0 = 2 * (x[0] - self.a[0]) - 4 * self.a[1] * x[0] * (x[1] - x[0] ** 2)
        dx1 = 2 * self.a[1] * (x[1] - x[0] ** 2)

        return jnp.array([dx0, dx1])

def newton_raphson(residual, initial_guess, tolerance=1e-8, max_iterations=100):
    """Find the root of a function using the Newton-Raphson method."""
   
    def body_function(i, state):
        _, _, current_solution, iteration = state
        residual_value = residual(current_solution)
        jacobian = jax.jacfwd(residual)(current_solution)
        next_solution = current_solution - jnp.linalg.solve(jacobian, residual_value)
        return (residual_value, jacobian, next_solution, iteration + 1)

    state_init = (
        jnp.full_like(initial_guess, 100),
        jnp.zeros(
            initial_guess.shape[:-1] + (initial_guess.shape[-1], initial_guess.shape[-1])
        ),
        initial_guess,
        0
    )
    _, jacobian_at_solution, final_solution, _ = jax.lax.fori_loop(
       0, max_iterations, body_function, state_init
    )

    return final_solution, jacobian_at_solution

def x_star(a): 
    rosenbrock = rosenbrock_func(a)
    initial_guess = jnp.array([0.0, 0.0])
    x_star, _ =  newton_raphson(rosenbrock.gradient, initial_guess)
    return x_star

def loss(x_star): 
    return jnp.linalg.norm(x_star)


def implicit_fwd(a): 
    rosenbrock = rosenbrock_func(a)
    initial_guess = jnp.array([0.0, 0.0])
    x_star, jacobian_sol =  newton_raphson(rosenbrock.gradient, initial_guess)
    return x_star, (x_star, jacobian_sol, a)



#Outcomment wrapper for profiling 
#@profile 
def backwards(a): 
    return jax.jacrev(lambda a: loss(x_star(a)))(a)

if __name__ == "__main__":
    sol = x_star([1, 100])
    print("x*:", sol)
    a = jnp.array([1.0, 100.0])
    
    time_1 = timeit.timeit(
            lambda: backwards(a), 
            number = 1
     )
    print(f"First method took {time_1:.6f} seconds" ) 









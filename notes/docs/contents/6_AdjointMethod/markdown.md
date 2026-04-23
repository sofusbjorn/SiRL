# Adjoint Method

Learning Objectives:

* Describe the challenges of evaluating gradients in solving initial value problem in a numerical optimization algorithm.

* Derive the calculation of gradient in adjoint method with implicit function theorem. Explain why this helps alleviating the challenges identified above.

* 

* Implement adjoint method in the loop of trajectory optimization problem with autodiff and linear equation solver libraries.

Possible assignment:

Implement trajectory optimization for the three-link reaching problem in your dynamics model or favourate differentiable simulation with adjoint method. Compare and discuss the difference between results obtained from adjoint method and auto-differentiation-based BPTT. (I guess we need setting some scenes to break autodiff/BPTT and highlight the benefits of adjoint method. Would it be an option to use larger time interval to blow up forward simulation while this is admissable with implicit solver? I am not very familiar if this will also make solving the linear system ill-posed or not. Will there be efficiency distinction? My feeling is that solving the linear equation can also be very costly unless sparse solving techniques were used?) Jax has an implmented feature!!!

```
**Dependencies on previous assignment**:

The optimization code and result from Optimal Control session: to plug the gradient now with ajoint method and do comparison
````
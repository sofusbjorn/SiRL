# Auto-differentiation and Program Transformation
## Gradient-based Optimization
Motivate optimization and evaluation of gradients; 

Describe some "naive" ways like finite-difference and symbolic derivation

Use Rosenbrock function as example
## Automatic-differentiation - Forward Mode
What is auto-differentiation: transforming a program that calculate numerical values to another one that compute both values and derivatives with the same accuracy

Use a arithematic function example and talk about its computing graph

Show forward-differentiation can be implemented together with the original code structure with extra variables

Talk about the superiority of such implementation and downside when the size of intermdiate variables are large
## Auto-differentiation - Reverse Mode

## Practical Consideration
Forward Mode or Reverse Mode

Memory management and higher order?: this could be just a little to attend the students about the applications they want to put AD on

Pytorch or Jax usage



## Learning objectives:

* Describe how auto-differentiation works under forward and reverse modes. 

* Explain the pros and cons of forward and reverse modes. 

* Account for how to choose a best performing mode for a given problem.

* Impelement gradient-based optimization with autodiff to solve the minimisation of Rosenbrock function.

* Explain the motivation of auto-differentiation and its difference from symbolic-based and finite-difference methods;


Possible assignment/exercise activities:

Implement the opt problem with a comparison of gradient evaluation between forward and reverse modes. Also compare to symbolic derivation and finite-difference for discussion on genericity and efficiency trade-off for these gradient evaluation methods.
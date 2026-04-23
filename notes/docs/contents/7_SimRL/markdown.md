# Simulation-based Reinforcement Learning 


## Learning objectives

* Explain what can be the challenges for applying a policy trained in simulation to real world;

* Explain how gradient-based optimization may help for calibrating the simulation so as to reduce the reality gap;

* Implement a real-to-sim process with gradient-based optimization to calibrate the length of links in the three-link model with mocap data;

* Implement a sim-to-real processes to control the selected three joints of UR5 with the calibration data to track a optitrack marker;



Possible assignment/exercise activities:

Use the data and auto-diff to estimate the link length of the three-link model; Compare with the actual length and discuss about the result.
Use the calibrated model for trajectory optimization; Compare to results obtained from RL and discuss pros and cons. 

````
**Dependencies on previous assignment**:

The dynamics model of three-link body they built or a simulation instance.
The scipy lib and autodiff they used: again solve the inverse problem but this time is the model param, e.g. link length
````
## References

# Robot Lab and Vision Data

* Describe how a mocap system works and the principle of optic-based systems;

* Describe how to turn on a UR5 robot arm and using the pendant to move the end-effector and emergency brake button;

* Account for safety rules and good lab behaviors in using mocap and robot systems;

* Implement code snippets to retrieve, filtering, visualise, store and reload mocap data;

* Implement python code snippets with UR RTDE library to control UR5 joint and tool center point motion and OnRobot RG2 gripper;

Possible assignment/exercise activities:

Attend lab training courses on safety and operation rules. Mount an optitracker at the end-effector of UR5; Set different position commands to three revolute joints (TODO: specify the ids) and collect corresponded joint configuration and tracker pose; 
<!-- Extra: use the data and auto-diff to estimate the link length of the three-link model; Compare with the actual length and discuss about the result. -->

```

Resources:

https://gitlab.com/sdurobotics/ur_rtde

https://github.com/UniversalRobots/RTDE_Python_Client_Library

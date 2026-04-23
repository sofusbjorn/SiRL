import os
os.environ.pop('SIRL_USE_JAX', None)
from articulated_dynamics.math_utils import nplib, rotate
from articulated_dynamics.dynamics import Dynamics, Kinematics
from articulated_dynamics.robots import ThreeLinks
from articulated_dynamics.integrator import integrate_euler

import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

class MyThreeLinksEnv(gym.Env):
    def __init__(self, link_len=0.3, link_den=2700) -> None:
        super().__init__()

        #define the state and action space
        #Box indicate a space defined by intervals with low/high bounds. Could be unbounded as well.
        self.observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))   
        #the continuous action space is recommended to have a range [-1, 1] for Gaussian noised NN output
        #see discussion in https://github.com/hill-a/stable-baselines/issues/678
        self.action_space=gym.spaces.Box(low=-30, high=30, shape=(3,))

        self.model = ThreeLinks(link_len=link_len, link_den=link_den)
        self.link_len = link_len
        self.joint_lim = [-7*nplib.pi/8, 7*nplib.pi/8]
        
        self.state = None
        self.t = 0
        self.horizon = 200

        self.goal = nplib.array([0.5, 0.5, 0])

    def reset(self, seed=None):
        super().reset(seed=seed)
        #reset the environment: sampling a new initial state according to /pho; and reset the clock

        self.state = np.zeros(6)
        #only randomize the angular position of links
        self.state[0] = 0.  #self.np_random.uniform(low=-np.pi/6, high=np.pi/6)
        self.state[1] = 0.  #self.np_random.uniform(low=-np.pi/6, high=np.pi/6)
        self.state[2] = 0.  #self.np_random.uniform(low=-np.pi/6, high=np.pi/6)

        self.t = 0
        self.dt = 0.01
        return self.get_obs(), {}
    
    def get_obs(self):
        return self.state #nplib.concatenate((self.state, self.calc_tippos(self.state)))
    
    def calc_tippos(self, state):
        x_pose = Kinematics._forward_pos(self.model, state[:3])[0][-1]
        #get tip position
        xp_tip = x_pose.trans + rotate(nplib.array([0, self.link_len/2, 0]), x_pose.rot)
        return xp_tip

    def reward(self, s, a):
        #try to reach a goal
        xp_tip = self.calc_tippos(s)
        return -nplib.linalg.norm(xp_tip - self.goal) - nplib.linalg.norm(a) * 1e-5 - nplib.linalg.norm(s[3:]) * 5e-4
    
    def step(self, action):
        #evaluate the reward
        r = self.reward(self.state, action)

        q = nplib.array(self.state[:3])
        qd = nplib.array(self.state[3:])

        tau = action 

        qdd = Dynamics.forward(self.model, q, qd, tau)

        #integrate for a small time step, 0.01s here, for the next state
        q_next, qd_next = integrate_euler(self.model, self.dt, q, qd, qdd, symplectic=False)

        #apply joint limits
        qd_next[q_next > self.joint_lim[1]] = 0.
        qd_next[q_next < self.joint_lim[0]] = 0.
        qd_next[qd_next > 5] = 5.
        qd_next[qd_next < -5] = -5.
        q_next[q_next > self.joint_lim[1]] = self.joint_lim[1]
        q_next[q_next < self.joint_lim[0]] = self.joint_lim[0]

        #type conversion to numpy state format
        self.state = np.concatenate([np.array(q_next), np.array(qd_next)])

        #tick the time step and check if the rollout has reached the end
        self.t += 1
        terminated = ( self.t >= self.horizon )
        #depending the gym version, should return the finishing signal as solely a Done or more concrete terminated or truncated
        #see https://farama.org/Gymnasium-Terminated-Truncated-Step-API
        #the last dictionary allows passing some extra info when needed
        return self.get_obs(), r, terminated, False, {}

if __name__ == "__main__":
    
    envs = SubprocVecEnv( [ lambda: MyThreeLinksEnv(link_len=0.3, link_den=2700 + (nplib.random.rand()-0.5)*3000) for _ in range(20) ] )

    model = SAC("MlpPolicy", envs, verbose=1, seed=0)
    model.learn(total_timesteps=500000)

    model.save("threelinks_reacher_dr")
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9 #6

        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        # Subtract target vector from pose - take absolute value as penalty score
        delta_penalty = abs((self.sim.pose[:3] - self.target_pos)).sum()
        z_penalty = 2* (abs(self.sim.pose[2] - self.target_pos[2]) / 300) - 1
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()  # PROVIDED CODE
        #reward = -((delta_penalty**2) / ((300*3)**2)) # Square penalty to increase penalty for larger vals - scale to roughly 0-1 range
        #reward = -(2*(delta_penalty/900**2)-1) # Rough scaling b/w -1 and 1
        reward = -( (2*(delta_penalty/900)-1) + (1.0*z_penalty)) #
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            # pose_all.append (self.sim.pose)    # only pose data in state
            
            # Add velocity info to x, y, z,Euler angles
            xyz_Euler_vel = np.concatenate((self.sim.pose, self.sim.v), axis=None)
   
            pose_all.append (xyz_Euler_vel)
            #print ('pose_all loop append: ', pose_all)
        next_state = np.concatenate(pose_all)
        #print (next_state, np.shape(next_state))
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # Add velocity info to x, y, z,Euler angles
        #state = np.concatenate([self.sim.pose] * self.action_repeat)
        xyz_Euler_vel =  np.concatenate((self.sim.pose, self.sim.v), axis=None)
        state = np.concatenate([xyz_Euler_vel] * self.action_repeat)
        return state
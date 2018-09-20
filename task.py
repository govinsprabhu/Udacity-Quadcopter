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
        self.init_pose = init_pose
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        #pos = np.array(self.sim.pose[:3])
        #target = np.array(self.target_pos[:3])
        #reward = np.tanh(2 - 4e-3 * np.sum((self.target_pos - self.sim.pose[2])**2)  - 3e-4 * np.sum(self.sim.v[2]))    
        #reward += self.sim.v[2]*10
        #done = False
        #if distance < 0.5: 
         #   done = True
        #reward = np.tanh(self.sim.pose[2] * 10 +self.sim.v[2] *10)
        #reward = np.tanh(1 + self.sim.v[2]  + self.sim.pose[2] )
        '''- np.sqrt(np.square(self.sim.v[:2]).sum())'''
        #reward = np.tanh(1 - 0.003 * (abs(self.target_pos - self.sim.pose[:3]))).sum()
        #np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        #dist_from_target = np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())
        # Want the quadcopter to hover straight up, measure angular deviation
        #angular_dev = np.sqrt(np.square(self.sim.pose[3:]).sum())
        # Also want to penalise movement not in z-axis (DeepMind Locomotion)
        #non_z_v = np.square(self.sim.v[:2]).sum()
        #reward = 1. - .001*dist_from_target - .0005*angular_dev - .0005*non_z_v
        #reward = np.tanh(1.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        reward = 0
        #if self.sim.pose[2] >= self.target_pos[2]:
        #    reward += 10
        #reward += np.tanh(np.linalg.norm(self.sim.pose[:3] -self.target_pos) + self.sim.v[2])
        # initial reward - fraction of the z-velocity
        reward = 0.5 * self.sim.v[2]
    
        # additional reward if the agent is close in vertical coordinates
        # to the target pose
        if abs(self.sim.pose[2] - self.target_pos[2]) < 3:
            reward += 15.
            done = True
        
        # penalize a crash
        if done and self.sim.time < self.sim.runtime: 
            reward = -1
            return reward, done
    
        # penalize the downward movement relative to the starting position
        if self.sim.pose[2] < self.init_pose[2]:
            reward -= 1

        return np.tanh((1 - .001 * np.linalg.norm(self.sim.pose[:3] - self.target_pos)) + reward + 0.1 * self.sim.v[2]), done
        
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward, done = self.get_reward(done)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state







    
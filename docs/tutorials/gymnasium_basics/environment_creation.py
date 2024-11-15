# Installation of Packages
pip install gym
pip install numpy
pip install pybullet  
pip install stable-baselines3  # DDPG, PPO (RL algorithms)

# New Dym Environment
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data

class CDPREnv(gym.Env):
    def __init__(self):
        super(CDPREnv, self).__init__()
        # Define action and observation space
        # Here we have 8 cables, so 8 actions corresponding to cable tensions
        self.action_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        
        # Observation space might include EE position, orientation, velocities, etc.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Initialize simulation
        p.connect(p.DIRECT)  # Use GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load CDPR and EE model (create a simple rectangular block for EE)
        self.reset()

# Set Function
    def reset(self):
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # Load EE as a simple rectangular block, adjust to model dimensions
        self.ee_id = p.loadURDF("block.urdf", [0, 0, 1])

        # Initialize cables (positions of attachment points) and other states
        self.cable_tensions = np.zeros(8)  # Start with zero tension
        return self._get_observation()

    def step(self, action):
        # Apply actions (adjust cable tensions)
        self.cable_tensions = np.clip(action, 0, 1)

        # Compute forces based on tensions (for simplicity, assume linear mapping)
        forces = self.cable_tensions * 100  # Scale forces appropriately

        # Apply forces on the EE
        for i in range(8):
            # Apply force on EE at specific attachment points
            p.applyExternalForce(self.ee_id, -1, forceObj=[0, 0, -forces[i]], posObj=[0, 0, 0], flags=p.WORLD_FRAME)

        # Step simulation
        p.stepSimulation()

        # Get observations
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)

        return obs, reward, done, {}

    def _get_observation(self):
        # Obtain position and orientation of EE
        position, orientation = p.getBasePositionAndOrientation(self.ee_id)
        velocity, angular_velocity = p.getBaseVelocity(self.ee_id)
        obs = np.array(position + orientation + velocity + angular_velocity)
        return obs

    def _compute_reward(self, obs):
        # Define a reward function (example: reward EE being near the target)
        target_position = np.array([1, 1, 1])
        position = obs[:3]
        reward = -np.linalg.norm(position - target_position)
        return reward

    def _check_done(self, obs):
        # Check if done (e.g., max steps or reaching close to the target)
        position = obs[:3]
        if np.linalg.norm(position - np.array([1, 1, 1])) < 0.05:
            return True
        return False

# Integration of Reinforcement Learning Algorithm (DDPG and PPO)
import gym
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_checker import check_env
from cdpr_env import CDPREnv

env = CDPREnv()
check_env(env)  # Optional: Check if env follows Gym API

# Choose an RL algorithm
model = DDPG("MlpPolicy", env, verbose=1)  # Or use PPO("MlpPolicy", env)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ddpg_cdpr")

# Test Running
python train_cdpr.py

# Training of the Algorithm
from stable_baselines3 import DDPG
from cdpr_env import CDPREnv

env = CDPREnv()
model = DDPG.load("ddpg_cdpr")

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()



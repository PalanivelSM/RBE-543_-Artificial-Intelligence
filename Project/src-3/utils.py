import random
import numpy as np
import torch
import glob
import io
import base64
import imageio
import gymnasium as gym
from IPython.display import HTML, display
from gymnasium.wrappers import RecordVideo
from copy import deepcopy


def show_video_of_model(agent, env_name, last_episode_actions=None):
    """
    Visualize the agent's performance in the environment.
    If last_episode_actions is provided, it will display that sequence of actions instead of dynamic actions.
    """
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frame = env.render()
        frames.append(frame)

        if last_episode_actions:
            action = last_episode_actions.pop(0)  # If action sequence is provided, use it
        else:
            action = agent.act(state)  # Otherwise, dynamically get action from the agent

        state, reward, done, _, _ = env.step(action.item() if isinstance(action, torch.Tensor) else action)

        # If action sequence is provided and exhausted, mark done even if the environment is not done
        if last_episode_actions is not None and len(last_episode_actions) == 0:
            done = True

    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def deepcopy_env(env):
    """Helper function to deepcopy the environment and ensure full isolation."""
    # Create a new instance of the environment
    new_env = gym.make(env.spec.id)
    new_env.reset()  # Reset the new environment to initialize its state

    # Copy attributes that are not automatically set by gym.make()
    for attr in dir(env.unwrapped):
        if not attr.startswith('__') and not callable(getattr(env.unwrapped, attr)):
            try:
                # Handle the SWIG-wrapped 'lander' object separately
                if attr == "lander":
                    original_lander = getattr(env.unwrapped, attr)
                    new_lander = recreate_lander(original_lander, new_env)
                    setattr(new_env.unwrapped, attr, new_lander)
                else:
                    setattr(new_env.unwrapped, attr, getattr(env.unwrapped, attr))
            except AttributeError:
                pass

    return new_env


def recreate_lander(original_lander, new_env):
    """Recreate the lander object in the new environment with the same attributes."""
    # We want an entirely new instance of the lander object in RAM, but we need to keep 
    # its inner attributes (e.g. position, velocity), which represent the state, the same.
    new_lander = new_env.unwrapped.lander
    new_lander.angle = original_lander.angle
    new_lander.angularVelocity = original_lander.angularVelocity
    new_lander.position.x = original_lander.position.x
    new_lander.position.y = original_lander.position.y
    # new_lander.inertia = original_lander.inertia

    return new_lander


# A class for recording and sampling a memory of experiences
class ReplayMemory(object):

  def __init__(self, capacity):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.capacity = capacity
    self.memory = []

  def push(self, event):
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size):
    experiences = random.sample(self.memory, k = batch_size)
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return states, next_states, actions, rewards, dones

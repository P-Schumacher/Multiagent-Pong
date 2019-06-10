import roboschool
import torch
from dqn_model import DQN
import wrappers
import time
import gym.wrappers

'''
This function is for seeing models in action, it uses a slightly modified version of the environment which has longer 
timeouts and longer episode lengths
'''

DEFAULT_ENV_NAME = "RoboschoolPong-v8"  # Use a longer version of Pong for demonstration (needs to be defined in source)
MAKE_VIDEO = False  # Set true or false here to record video OR render, not both

env = gym.make(DEFAULT_ENV_NAME)
env = wrappers.action_space_discretizer(env, 10)
net = DQN(env.observation_space.shape[0], env.action_space.n)
# net.load_state_dict(torch.load("RoboschoolPong-v1-best_var_batch.dat"))
# net.load_state_dict(torch.load("RoboschoolPong-v1-best_night_training.dat"))
# net.load_state_dict(torch.load("RoboschoolPong-v1-end_of_training.dat"))
net.load_state_dict(torch.load("RoboschoolPong-v1-time_update.dat"))
env.reset()
recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, "./recording.mp4", enabled=MAKE_VIDEO)

if not MAKE_VIDEO:
    env.render()

for i in range(1):
    obs = env.reset()
    while True:
        recorder.capture_frame()
        action = net(torch.tensor(obs, dtype=torch.float32)).max(0)[1]
        action = action.item()
        action = int(action)
        obs, reward, done, _ = env.step(action)
        if not MAKE_VIDEO:
            time.sleep(0.011)
        if done:
            break
recorder.close()

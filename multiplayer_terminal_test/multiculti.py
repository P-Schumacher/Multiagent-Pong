import gym, roboschool, sys
import numpy as np
import torch
from dqn_model import DQN
import wrappers
import gym.wrappers
import time

RECORD = False  # do you want to record?

def play(env, net, recorder, video):
    episode_n = 0
    while 1:
        episode_n += 1
        obs = env.reset()
        
        while 1:
            if not video: time.sleep(0.001)
            if video: recorder.capture_frame()
            action = net(torch.tensor(obs, dtype=torch.float32)).max(0)[1]
            action = action.item()
            action = int(action)
            obs, rew, done, info = env.step(action)
            if done: break
        if video: recorder.close()
        break



player_n = int(sys.argv[2])


env = gym.make("RoboschoolPong-v1")
env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)
net = DQN(13, 4)
net.load_state_dict(torch.load("RoboschoolPong-v1-best_night_training.dat"))
env = wrappers.action_space_discretizer(env, 2)

if RECORD and player_n == 0:
    video = True
else:
    video = False
recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, "./recording.mp4", enabled=video)
play(env, net, recorder, video)

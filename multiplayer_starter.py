import gym
import roboschool
import helpfunc
import helpclass
import sys
import roboschool.multiplayer



params = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,  # discount factor in Bellman update
              "BATCH_SIZE": 256,  # how many samples at the same time (has to be big for convergence of TD 1 step)
              "LOAD_PREVIOUS ": False,  # Set to true if we want to further train a previous model
              "REPLAY_SIZE": 10000,  # size of replay buffer
              "LEARNING_RATE": 1e-4,  # learning rate of neural network update
              "SYNC_TARGET_FRAMES": 10000,  # when to sync neural net and target network (low values destroy loss func)
              "REPLAY_START_SIZE": 10000,  # how much do we fill the buffer before training
              "EPSILON_DECAY_LAST_FRAME": 10000,  # how fast does the epsilon exploration decay
              "NUMBER_FRAMES": 300000,  # total number of training frames
              "ACTION_SIZE": 10,  # network doesnt seem to care much about action_space discretization...
              "SKIP_NUMBER": 4,  # how many frames are skipped with repeated actions != n step DQN
              "EPSILON_START": 1,
              "EPSILON_FINAL": 0.02,
              "device": "cpu"}


agent, buffer, optimizer, writer, net, tgt_net = helpfunc.start_env(params)

agent.env.unwrapped.multiplayer(agent.env, game_server_guid="selfplayer", player_n=int(sys.argv[1]))






import helpfunc
import dqn_model
import gym
import torch
import time
# TODO multiplayer training, automated hyperparameter search, double q learning, n step q learning

'''
Main function: Defines important constants, initializes all the important classes and does the training.
'''

params = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,  # discount factor in Bellman update
              "BATCH_SIZE": 256,  # how many samples at the same time (has to be big for convergence of TD 1 step)
              "LOAD_PREVIOUS ": True,  # Set to true if we want to further train a previous model
              "REPLAY_SIZE": 10000,  # size of replay buffer
              "LEARNING_RATE": 1e-4,  # learning rate of neural network update
              "SYNC_TARGET_FRAMES": 1000,  # when to sync neural net and target network (low values destroy loss func)
              "REPLAY_START_SIZE": 10000,  # how much do we fill the buffer before training
              "EPSILON_DECAY_LAST_FRAME": 10000,  # how fast does the epsilon exploration decay
              "NUMBER_FRAMES": 300000,  # total number of training frames
              "ACTION_SIZE": 3,  # network doesnt seem to care much about action_space discretization...
              "SKIP_NUMBER": 4,  # how many frames are skipped with repeated actions != n step DQN
              "EPSILON_START": 1,
              "EPSILON_FINAL": 0.02,
              "device": "cpu",
              "double": True,
              "selfplay": True,
              "player_n": None,
              "selfsync": 1000}

# cpu faster than cuda, network is so small that the time needed to load it into the gpu is larger than
# the gained time of parallel computing


def play(obs, net):
    action = net(torch.tensor(obs, dtype=torch.float32)).max(0)[1]
    action = action.item()
    action = int(action)
    return action


def multiplayer_agent_player_0(params):
    params["player_n"] = 0
    reward = helpfunc.train(params)
    return reward


def multiplayer_agent_player_1(params, n):
    params["player_n"] = n
    env = gym.make(params["DEFAULT_ENV_NAME"])
    if params["selfplay"]:
        env.unwrapped.multiplayer(env, game_server_guid="selfplayer", player_n=params["player_n"])
    env = helpfunc._construct_env(env, params["ACTION_SIZE"], params["SKIP_NUMBER"])
    net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    n = 0
    net.load_state_dict(torch.load("ddqn.dat"))
    while 1:
        obs = env.reset()
        while 1:
            # if (n % params["selfsync"]) == 0:
            #     net.load_state_dict(torch.load("RoboschoolPong-v1-time_update.dat"))
            n += 1
            action = play(obs, net)
            obs, reward, done, _ = env.step(action)
            if done:
                break
    return reward


if __name__ == '__main__':
    if params["selfplay"]:
        import roboschool.multiplayer
        import torch.multiprocessing as mp
        game = roboschool.gym_pong.PongSceneMultiplayer()
        gameserver = roboschool.multiplayer.SharedMemoryServer(game, "selfplayer", want_test_window=False)
        player_0 = mp.Process(target=multiplayer_agent_player_0, args=(params,))
        player_1 = mp.Process(target=multiplayer_agent_player_1, args=(params, 1))

        player_0.start()
        player_1.start()

        gameserver.serve_forever()

        player_0.join()
        player_1.join()
    else:
        performance = helpfunc.train(params)

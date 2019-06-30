import helpfunc
import dqn_model
import gym
import torch
import time
# TODO automated hyperparameter search, n step q learning

'''
Main function: Defines important constants, initializes all the important classes and does the training.
'''

params = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,  # discount factor in Bellman update
              "BATCH_SIZE": 256,  # how many samples at the same time (has to be big for convergence of TD 1 step)
              "LOAD_PREVIOUS ": False,  # Set to true if we want to further train a previous model
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
              "player_n": 0}

# cpu faster than cuda, network is so small that the time needed to load it into the gpu is larger than
# the gained time of parallel computing


def multiplayer_agent_player(params, net, player_number):
    params["player_n"] = player_number
    reward = helpfunc.train(params, net)
    return reward


if __name__ == '__main__':
    if params["selfplay"]:
        import roboschool.multiplayer
        import torch.multiprocessing as mp
        game = roboschool.gym_pong.PongSceneMultiplayer()
        gameserver = roboschool.multiplayer.SharedMemoryServer(game, "selfplayer", want_test_window=False)

        mp.set_start_method('spawn')
        env_tmp = gym.make(params["DEFAULT_ENV_NAME"])
        net = dqn_model.DQN(env_tmp.observation_space.shape[0], params["ACTION_SIZE"]**2).to(device=params["device"])
        '''
        share_memory() moves the NN to the system shared memory, it can then be accessed by both processes. 
        This is not necessary when using cuda, because the GPU memory is shared by default. It becomes a no-op on GPU.
        '''
        net.share_memory()
        player_0 = mp.Process(target=multiplayer_agent_player, args=(params, net, 0))
        player_1 = mp.Process(target=multiplayer_agent_player, args=(params, net, 1))

        player_0.start()
        player_1.start()
        try:  # serve_forever() runs forever until it crashes or we interrupt. (try: finally:) closes the subprocesses
            gameserver.serve_forever()
        finally:
            player_0.join()
            player_1.join()
    else:
        env_tmp = gym.make(params["DEFAULT_ENV_NAME"])
        net = dqn_model.DQN(env_tmp.observation_space.shape[0], params["ACTION_SIZE"] ** 2).to(device=params["device"])
        performance = helpfunc.train(params, net)

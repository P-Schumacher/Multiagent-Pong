import helpfunc

# TODO multiplayer training, automated hyperparameter search, double q learning, n step q learning

'''
Main function: Defines important constants, initializes all the important classes and does the training.
'''

params = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,  # discoutn factor in Bellman update
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

# cpu faster than cuda, network is so small that the time needed to load it into the gpu is larger than
# the gained time of parallel computing





if __name__ == '__main__':
    reward = helpfunc.train(params)
    print("The mean episode reward over the last %i episodes was " %params["EPSILON_DECAY_LAST_FRAME"], reward)

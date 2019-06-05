import wrappers
import project_classes
import dqn_model
import torch
from torch import optim
import gym
import roboschool
from tensorboardX import SummaryWriter
import time
from matplotlib import pyplot as plt


'''
Main function: Defines important constants, initializes all the important classes and does the training.
Will be made prettier
'''

DEFAULT_ENV_NAME = "RoboschoolPong-v1"
LOAD_PREVIOUS = False  # Set to true if we want to further train a previous model

GAMMA = 0.99
BATCH_SIZE = 256

REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 10000
REPLAY_START_SIZE = 10000


NUMBER_FRAMES = 1000000
EPSILON_DECAY_LAST_FRAME = 400000
if LOAD_PREVIOUS:
    EPSILON_START = 0.02
else:
    EPSILON_START = 1
EPSILON_FINAL = 0.02

device = "cpu"  # actually faster than cuda.... I blame the Geforce 850M

if __name__ == '__main__':

    # ______________________________CREATE AND INITIALIZE OBJECTS_______________________________
    env = gym.make(DEFAULT_ENV_NAME)
    # Discretize the action space and give it n*n actions, works better in DQN than high dimensional action_spaces
    env = wrappers.action_space_discretizer(env, n=3)
    env = wrappers.SkipEnv(env, skip=4)
    buffer = project_classes.ExperienceBuffer(REPLAY_SIZE)
    agent = project_classes.Agent(env, buffer)

    net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(device)

    if LOAD_PREVIOUS:  # Set path to previous model here
        net.load_state_dict(torch.load("RoboschoolPong-v1-best_var_batch.dat"))
        tgt_net.load_state_dict(torch.load("RoboschoolPong-v1-best_var_batch.dat"))


    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_reward = []

    # ______________________________POPULATE REPLAY BUFFER_______________________________

    env.reset()
    # env.render()
    print("Populating Buffer...")
    for i in range(REPLAY_START_SIZE):
        agent.play_step(net, EPSILON_START, device)
    batch = buffer.sample(BATCH_SIZE)
    print("Buffer populated!")

    print("Start training:")

    # ______________________________TRAINING__________________________________________________

    best_reward = - 1000
    start_time = time.time()
    for i in range(NUMBER_FRAMES):
        # if i == 100000 and not LOAD_PREVIOUS:
        #     BATCH_SIZE = 128
        # if i == 200000 and not LOAD_PREVIOUS:
        #     BATCH_SIZE = 256
        epsilon = max(EPSILON_FINAL, EPSILON_START - i / EPSILON_DECAY_LAST_FRAME)
        ep_reward = agent.play_step(net, epsilon, device)
        if ep_reward:
            writer.add_scalar("episode_reward", ep_reward, i)
            if ep_reward > best_reward:
                best_reward = ep_reward
                writer.add_scalar("best reward", best_reward)
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")

        if (i % SYNC_TARGET_FRAMES) == 0:
            tgt_net.load_state_dict(net.state_dict())
            print("We are at: %i / %i frames" % (i, NUMBER_FRAMES))
            print("At: %f s" % (time.time() - start_time))
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = project_classes.calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        writer.add_scalar("loss", loss_t, i)
        writer.add_scalar("epsilon", epsilon, i)
        optimizer.step()

    writer.close()
    print("End training!")
    print("Time was: %f" % (time.time() - start_time))

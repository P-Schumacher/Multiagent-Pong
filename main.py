import wrappers
import project_classes
import dqn_model
import torch
from torch import optim
import gym
import roboschool
from tensorboardX import SummaryWriter
import time


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

NUMBER_FRAMES = 5000000
EPSILON_DECAY_LAST_FRAME = 50000
if LOAD_PREVIOUS:
    EPSILON_START = 0.02
else:
    EPSILON_START = 1
EPSILON_FINAL = 0.02

device = "cpu"
# cpu faster than cuda, network is so small that the time needed to load it into the gpu is larger than
# the gained time of parallel computing


if __name__ == '__main__':

    # ______________________________CREATE AND INITIALIZE OBJECTS_______________________________
    env = gym.make(DEFAULT_ENV_NAME)
    env = wrappers.action_space_discretizer(env, n=2)
    env = wrappers.SkipEnv(env, skip=4)
    buffer = project_classes.ExperienceBuffer(REPLAY_SIZE)
    agent = project_classes.Agent(env, buffer)

    net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(device)

    if LOAD_PREVIOUS:  # Set path to previous model here
        net.load_state_dict(torch.load("RoboschoolPong-v1-best_var_batch.dat"))
        tgt_net.load_state_dict(torch.load("RoboschoolPong-v1-best_var_batch.dat"))

    writer = SummaryWriter(comment="-" + "batch" + str(BATCH_SIZE) + "_n" + str(env.action_space.n) + "_eps"
                                   + str(EPSILON_DECAY_LAST_FRAME) + "_skip" + str(4))
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # ______________________________POPULATE REPLAY BUFFER_______________________________

    env.reset()
    # env.render()
    print("Populating Buffer...")
    for i in range(REPLAY_START_SIZE):
        agent.play_step(net, EPSILON_START, device)
    batch = buffer.sample(BATCH_SIZE)
    print("Buffer populated!")

    # ______________________________TRAINING__________________________________________________
    print("Start training: ")
    best_reward = - 1000  # Initialize at a very low value
    start_time = time.time()
    for frame in range(NUMBER_FRAMES):
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame / EPSILON_DECAY_LAST_FRAME)
        ep_reward = agent.play_step(net, epsilon, device)
        if ep_reward:
            writer.add_scalar("episode_reward", ep_reward, frame)
            if ep_reward > best_reward:
                best_reward = ep_reward
                writer.add_scalar("best reward", best_reward, frame)
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")

        if (frame % SYNC_TARGET_FRAMES) == 0:
            tgt_net.load_state_dict(net.state_dict())
            print("We are at: %i / %i frames" % (frame, NUMBER_FRAMES))
            torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-time_update.dat")
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = project_classes.calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        writer.add_scalar("loss", loss_t, frame)
        writer.add_scalar("epsilon", epsilon, frame)
        optimizer.step()

    writer.close()
    print("End training!")
    print("Time was: %f" % (time.time() - start_time))

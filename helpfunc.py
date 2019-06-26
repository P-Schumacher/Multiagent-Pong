import numpy as np
import wrappers
import helpclass
import dqn_model
import gym
import roboschool
import torch
from tensorboardX import SummaryWriter
from torch import optim

# implemented in pongsimulation.py: create_simulation
def start_env(params):
    agent, buffer, net, tgt_net = _setup_all(params)

    fill_buffer(agent, params["REPLAY_START_SIZE"], params["REPLAY_SIZE"])

    load_model(net, tgt_net, "RoboschoolPong-v1-best.dat", params["LOAD_PREVIOUS "])

    optimizer = optim.Adam(net.parameters(), lr=params["LEARNING_RATE"])

    writer = SummaryWriter(comment="-" + "batch" + str(params["BATCH_SIZE"]) + "_n" + str(agent.env.action_space.n) +
                                   "_eps" + str(params["EPSILON_DECAY_LAST_FRAME"]) + "_skip" + str(4) + "learning_rate"
                                   + str(params["LEARNING_RATE"]))

    if params["LOAD_PREVIOUS "]:
        params["EPSILON_START"] = params["EPSILON_FINAL"]

    return agent, buffer, optimizer, writer, net, tgt_net

# implemented in pongsimulation.py: create_environment, create_agent, create_model
def _setup_all(params):
    env = gym.make(params["DEFAULT_ENV_NAME"])
    env = _construct_env(env, params["ACTION_SIZE"], params["SKIP_NUMBER"])
    buffer = helpclass.ExperienceBuffer(params["REPLAY_SIZE"])
    agent = helpclass.Agent(env, buffer)
    net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    tgt_net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    return agent, buffer, net, tgt_net

# implemented in pongsimulation.py: create_environment
def _construct_env(env, n_actions, n_skip):
    env = wrappers.action_space_discretizer(env, n=n_actions)
    env = wrappers.SkipEnv(env, skip=n_skip)
    return env

# implemented in buffers.py
def fill_buffer(agent, start_size, replay_size):
    assert start_size <= replay_size, "Start size of buffer is bigger than buffer size!"
    state = agent.env.reset()
    print("Populating Buffer...")
    for frames in range(start_size):
        action = agent.env.action_space.sample()
        new_state, reward, is_done, _ = agent.env.step(action)
        exp = helpclass.Experience(state, action, reward, is_done, new_state)
        agent.exp_buffer.append(exp)
    print("Buffer populated!")

# implemented in pongsimulation.py
def load_model(net, tgt_net, file_name=None, activator=False):
    if file_name is not None:
        assert type(file_name) == str, "Name of model to be loaded has to be a string!"
    if not activator:
        return None
    else:
        net.load_state_dict(torch.load(file_name))
        tgt_net.load_state_dict(torch.load(file_name))
        return None

def train(params):
    agent, buffer, optimizer, writer, net, tgt_net = start_env(params)
    # ______________________________TRAINING__________________________________________________
    print("Start training: ")
    best_reward = - 1000  # Initialize at a very low value
    reward = []
    for frame in range(params["NUMBER_FRAMES"]):
        epsilon = max(params["EPSILON_FINAL"], params["EPSILON_START"] - frame / params["EPSILON_DECAY_LAST_FRAME"])
        ep_reward = agent.play_step(net, epsilon, params["device"])
        if ep_reward:
            reward.append(ep_reward)
            writer.add_scalar("episode_reward", ep_reward, frame)
            writer.add_scalar("mean100_reward", np.mean(reward[-100:]), frame)

            if ep_reward > best_reward:
                best_reward = ep_reward
                writer.add_scalar("best reward", best_reward, frame)
                torch.save(net.state_dict(), params["DEFAULT_ENV_NAME"] + "-best.dat")

        if (frame % params["SYNC_TARGET_FRAMES"]) == 0:
            tgt_net.load_state_dict(net.state_dict())  # Syncs target and Standard net
            print("We are at: %7i / %7i frames" % (frame, params["NUMBER_FRAMES"]))
            torch.save(net.state_dict(), params["DEFAULT_ENV_NAME"] + "-time_update.dat")

        optimizer.zero_grad()
        batch = buffer.sample(params["BATCH_SIZE"])
        loss_t = helpclass.calc_loss(batch, net, tgt_net, params["GAMMA"], params["device"])
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t, frame)
        writer.add_scalar("epsilon", epsilon, frame)

    writer.close()
    print("End training!")
    torch.save(net.state_dict(), params["DEFAULT_ENV_NAME"] + "end_of_training.dat")
    return np.mean(reward[:params["EPSILON_DECAY_LAST_FRAME"]])
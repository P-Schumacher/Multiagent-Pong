# can be filled at a later date with convenient functions which make
# the main function more high-level
# will become important when automating parameter search
import wrappers
import helpclass
import dqn_model
import gym
import roboschool
import torch


def setup_all(params):
    env = gym.make(params["DEFAULT_ENV_NAME"])
    env = construct_env(env, params["ACTION_SIZE"], params["SKIP_NUMBER"])
    buffer = helpclass.ExperienceBuffer(params["REPLAY_SIZE"])
    agent = helpclass.Agent(env, buffer)
    net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    tgt_net = dqn_model.DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    return buffer, agent, net, tgt_net


def construct_env(env, n_actions, n_skip):
    env = wrappers.action_space_discretizer(env, n=n_actions)
    env = wrappers.SkipEnv(env, skip=n_skip)
    return env


def load_buffer(agent, start_size, replay_size):
    assert start_size <= replay_size, "Start size of buffer is bigger than buffer size!"
    state = agent.env.reset()
    print("Populating Buffer...")
    for frames in range(start_size):
        action = agent.env.action_space.sample()
        new_state, reward, is_done, _ = agent.env.step(action)
        exp = helpclass.Experience(state, action, reward, is_done, new_state)
        agent.exp_buffer.append(exp)
    print("Buffer populated!")


def load_previous_model(net, tgt_net, file_name=None, activator=False):
    if file_name is not None:
        assert type(file_name) == str, "Name of model to be loaded has to be a string!"
    if not activator:
        return None
    else:
        net.load_state_dict(torch.load(file_name))
        tgt_net.load_state_dict(torch.load(file_name))
        return None


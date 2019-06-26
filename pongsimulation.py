import wrappers
import gym
import pongagent
from buffers import ExperienceBuffer
from dqn_model import DQN

def create_simulation(params):
    env = create_environment(params["DEFAULT_ENV_NAME"], params["ACTION_SIZE"], params["SKIP_NUMBER"])
    agent = create_agent(env, params["REPLAY_SIZE"])
    net = DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    tgt_net = DQN(env.observation_space.shape[0], env.action_space.n).to(params["device"])
    return agent, net, tgt_net

def create_environment(name, n_actions, n_skip):
    env = gym.make(name)
    env = wrappers.action_space_discretizer(env, n=n_actions)
    env = wrappers.SkipEnv(env, skip=n_skip)
    return env

def create_agent(env, nsamples):
    buffer = ExperienceBuffer(nsamples)
    agent = pongagent.Agent(env, buffer)
    agent.exp_buffer.fill(nsamples)
    return agent
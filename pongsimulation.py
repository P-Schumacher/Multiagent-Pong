import wrappers
import gym
import torch
import numpy as np
import pongagent
from buffers import ExperienceBuffer
from dqn_model import DQN, calc_loss
from torch import optim
from tensorboardX import SummaryWriter


# TODO: split parameters into simulation parameters and training parameters to pass to run?
class Simulation:
    """
    Simulation for the game of 3D Pong.

    Parameters
    ----------
    params: dict
            Dictionary of all the simulation parameters
    """
    def __init__(self, params):
        # unpack the parameters:
        #### simulation
        self.device = params["device"]
        self.env_name = params["env_name"]
        self.training_frames = params["training_frames"]
        self.skip_frames = params["skip_frames"]
        self.nactions = params["nactions"]
        self.messages_enabled = params["messages_enabled"]
        #### qnet model
        self.learning_rate = params["learning_rate"]
        self.sync = params["sync"]
        self.load_from = params["load_from"]
        #### buffer
        self.batch_size = params["batch_size"]
        self.replay_size = params["replay_size"]
        #### agent model
        self.gamma = params["gamma"]
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_decay_rate = params["eps_decay_rate"]

        # initialize the simulation
        self.env = self._create_environment()
        self.agent = self._create_agent()
        self.net, self.tgt_net, self.optimizer = self._create_model()
        self.writer = SummaryWriter(comment="-" + "batch" + str(self.batch_size)
                                            + "_n" + str(self.env.action_space.n)
                                            + "_eps" + str(self.eps_decay_rate)
                                            + "_skip" + str(self.skip_frames)
                                            + "learning_rate" + str(self.learning_rate))

    def _create_environment(self):
        """
        create a gym environment for the simulation.

        Actions are discretized into nactions and frames are skipped for faster training
        :return: env
        """
        env = gym.make(self.env_name)
        env = wrappers.action_space_discretizer(env, n=self.nactions)
        env = wrappers.SkipEnv(env, skip=self.skip_frames)
        return env

    def _create_agent(self):
        """
        Create agent with buffer for the simulation.

        :return: agent
        """
        buffer = ExperienceBuffer(self.replay_size)
        agent = pongagent.Pongagent(self.env, buffer)
        if self.messages_enabled:
            print("Populating Buffer ...")
        agent.exp_buffer.fill(self.env)
        if self.messages_enabled:
            print("Buffer_populated!")
        return agent

    def _create_model(self):
        """
        Create a deep Q model for function approximation with Adam optimizer.

        :return: net, tgt_net, optimizer
        """
        net = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        tgt_net = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        if self.load_from is not None:
            assert type(self.load_from) == str, "Name of model to be loaded has to be a string!"
            net.load_state_dict(torch.load(self.load_from))
            tgt_net.load_state_dict(torch.load(self.load_from))
        optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
        return net, tgt_net, optimizer

# TODO: clean this function!
    def run(self, mode="play"):
        """
        runs the simulation.
        :param mode: str, either "play" or "train"
        :return: mean reward over all episodes with eps_end
        """
        if mode == "train":
            if self.messages_enabled:
                print("Start training: ")
            best_reward = - 1000  # Initialize at a very low value
            reward = []
            for frame in range(self.training_frames):
                epsilon = max(self.eps_end, self.eps_start - frame / self.eps_decay_rate)
                ep_reward = self.agent.play_step(self.net, epsilon, self.device)
                if ep_reward:
                    reward.append(ep_reward)
                    self.writer.add_scalar("episode_reward", ep_reward, frame)
                    self.writer.add_scalar("mean100_reward", np.mean(reward[-100:]), frame)

                    if ep_reward > best_reward:
                        best_reward = ep_reward
                        self.writer.add_scalar("best reward", best_reward, frame)
                        torch.save(self.net.state_dict(), self.env_name + "-best.dat")

                if (frame % self.sync) == 0:
                    self.tgt_net.load_state_dict(self.net.state_dict())  # Syncs target and Standard net
                    if self.messages_enabled:
                        print("We are at: %7i / %7i frames" % (frame, self.training_frames))
                    torch.save(self.net.state_dict(), self.env_name + "-time_update.dat")

                self.optimizer.zero_grad()
                batch = self.agent.exp_buffer.sample(self.batch_size)
                loss_t = calc_loss(batch, self.net, self.tgt_net, self.gamma, self.device)
                loss_t.backward()
                self.optimizer.step()

                self.writer.add_scalar("loss", loss_t, frame)
                self.writer.add_scalar("epsilon", epsilon, frame)

            self.writer.close()
            if self.messages_enabled:
                print("End training!")
            torch.save(self.net.state_dict(), self.env_name + "end_of_training.dat")

            return np.mean(reward[-len(reward)//2:])

        elif mode == "play":
            assert type(self.load_from) == str, "Name of model to load not correctly given"
            pass

        else:
            print("Mode unknown")

import wrappers
import gym
import torch
import numpy as np
import pongagent
from buffers import Extendedbuffer
from buffers import ExperienceBuffer
from dqn_model import DQN, calc_loss
from torch import optim
from tensorboardX import SummaryWriter
import roboschool.multiplayer
import torch.multiprocessing as mp

# TODO: split parameters into simulation parameters and training parameters to pass to run?
class Simulation:
    """
    Simulation for the game of 3D Pong.

    Parameters
    ----------
    params: dict
            Dictionary of all the simulation parameters
    """
    def __init__(self, params, player_n = 0):
        # unpack the parameters:
        #### simulation
        self.device = params["device"]
        self.env_name = params["env_name"]
        self.training_frames = params["training_frames"]
        self.skip_frames = params["skip_frames"]
        self.nactions = params["nactions"]
        self.messages_enabled = params["messages_enabled"]
        self.selfplay = params["selfplay"]
        #### qnet model
        self.learning_rate = params["learning_rate"]
        self.sync = params["sync"]
        self.load_from = params["load_from"]
        #### buffer
        self.batch_size = params["batch_size"]
        self.replay_size = params["replay_size"]
        self.nstep = params["nstep"]
        #### agent model
        self.gamma = params["gamma"]
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_decay_rate = params["eps_decay_rate"]
        self.player_n = player_n
        self.double = params["double"]
        # initialize the simulation with shared properties
        self.env = gym.make(self.env_name)   # environment, agent etc. can"t be created jointly in a server simulation
        self.net = DQN(self.env.observation_space.shape[0], self.nactions**2).to(self.device)

    def _create_environment(self):
        """
            create a gym environment for the simulation.

            Actions are discretized into nactions and frames are skipped for faster training
            :return: env
            """
        env = gym.make(self.env_name)
        if self.selfplay:
            env.unwrapped.multiplayer(env, game_server_guid="selfplayer", player_n=self.player_n)
        env = wrappers.action_space_discretizer(env, n=self.nactions)
        env = wrappers.SkipEnv(env, skip=self.skip_frames)
        return env

    def _create_agent(self, env):
        """
            Create agent with buffer for the simulation.

            :return: agent
            """
        # buffer = ExperienceBuffer(self.replay_size)
        buffer = Extendedbuffer(self.replay_size, nstep=self.nstep, gamma=self.gamma)
        agent = pongagent.Pongagent(env, self.player_n, buffer)
        return agent

    def _create_model(self):
        """
            Create a deep Q model for function approximation with Adam optimizer.

            :return: net, tgt_net, optimizer
            """
        tgt_net = DQN(self.env.observation_space.shape[0], self.nactions**2).to(self.device)
        if self.load_from is not None:
            assert type(self.load_from) == str, "Name of model to be loaded has to be a string!"
            self.net.load_state_dict(torch.load(self.load_from))
            tgt_net.load_state_dict(torch.load(self.load_from))
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return tgt_net, optimizer

    def _init_non_shared(self, player_n):
        env = self._create_environment()
        tgt_net, optimizer = self._create_model()
        agent = self._create_agent(env)
        writer = SummaryWriter(comment="-"
                                       + "player" + str(player_n)
                                       + "batch" + str(self.batch_size)
                                       + "_n" + str(env.action_space.n)
                                       + "_eps" + str(self.eps_decay_rate)
                                       + "_skip" + str(self.skip_frames)
                                       + "learning_rate" + str(self.learning_rate))
        return env, agent, tgt_net, optimizer, writer

    def _fill_buffer(self, agent):
        if self.messages_enabled:
            print("Player populating Buffer ...")
        agent.exp_buffer.fill(agent.env, self.replay_size, self.nstep)
        if self.messages_enabled:
            print("Buffer_populated!")

    def train(self, net, player_n=0):
        self.net = net
        env, agent, tgt_net, optimizer, writer = self._init_non_shared(player_n)
        self._fill_buffer(agent)
        if self.messages_enabled:
            print("Player %i start training: " %player_n)
        reward = []
        for frame in range(self.training_frames):
            epsilon = max(self.eps_end, self.eps_start - frame / self.eps_decay_rate)
            ep_reward = agent.play_step(net, epsilon, self.device)
            if ep_reward:
                reward.append(ep_reward)
                writer.add_scalar("episode_reward", ep_reward, frame)
                writer.add_scalar("mean100_reward", np.mean(reward[-100:]), frame)
            if (frame % self.sync) == 0:
                tgt_net.load_state_dict(net.state_dict())  # Syncs target and Standard net
                if self.messages_enabled:
                    print("We are at: %7i / %7i frames" % (frame, self.training_frames))
                if player_n == 0:
                    torch.save(net.state_dict(), self.env_name + "-time_update.dat")

            optimizer.zero_grad()
            batch = agent.exp_buffer.sample(self.batch_size)
            loss_t = calc_loss(batch, net, tgt_net, self.gamma**self.nstep, self.double, self.device)
            loss_t.backward()
            optimizer.step()

            writer.add_scalar("loss", loss_t, frame)
            writer.add_scalar("epsilon", epsilon, frame)

        writer.close()
        if self.messages_enabled:
            print("Player %i end training!" %player_n)
        torch.save(net.state_dict(), self.env_name + "end_of_training.dat")

        return np.mean(reward[-len(reward)//2:])

    # TODO: clean this function!
    def run(self, mode="play"):
        """
        runs the simulation.
        :param mode: str, either "play" or "train"
        :return: mean reward over all episodes with eps_end
        """
        if mode == "train":
            reward = self.train(self.net)
            return reward
        elif mode == "play":
            pass

        else:
            raise Exception("Mode should be either play or train")





class PongSelfplaySimulation(Simulation):

    def __init__(self, params):
        self.params = params
        super(PongSelfplaySimulation, self).__init__(params)
        game = roboschool.gym_pong.PongSceneMultiplayer()
        self.gameserver = roboschool.multiplayer.SharedMemoryServer(game, "selfplayer", want_test_window=False)
        try:
            mp.set_start_method('spawn')
        except:
            pass
        self.net.share_memory()

    def run(self, mode="train"):
        self.net.share_memory()
        sim0 = Simulation(self.params, 0)
        sim1 = Simulation(self.params, 1)
        player_0 = mp.Process(target=sim0.train, args=(self.net, 0,))
        player_1 = mp.Process(target=sim1.train, args=(self.net, 1,))
        player_0.start()
        player_1.start()
        try:
            self.gameserver.serve_forever()
        finally:
            player_0.join()
            player_1.join()



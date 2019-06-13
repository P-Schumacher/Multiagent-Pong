import helpfunc
import helpclass
import torch
import numpy as np


# TODO multiplayer training (self play), automated hyperparameter search, double q learning, n step q learning

'''
Main function: Defines important constants, initializes all the important classes and does the training.
'''

params = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,  # discount factor in Bellman update
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

    # ______________________________PREPARE AGENT ENVIRONMENT, BUFFER, NETS _______________________________
    agent, buffer, optimizer, writer, net, tgt_net = helpfunc.start_env(params)

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
            print("We are at: %i / %i frames" % (frame, params["NUMBER_FRAMES"]))
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

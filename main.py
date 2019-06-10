import helper_functions
import project_classes
import torch
from torch import optim
import numpy as np
from tensorboardX import SummaryWriter


'''
Main function: Defines important constants, initializes all the important classes and does the training.
Will be made prettier
'''

Parameters = {"DEFAULT_ENV_NAME": "RoboschoolPong-v1",
              "GAMMA": 0.99,
              "BATCH_SIZE": 256,
              "LOAD_PREVIOUS ": False,  # Set to true if we want to further train a previous model
              "REPLAY_SIZE": 10000,
              "LEARNING_RATE": 0.02,
              "SYNC_TARGET_FRAMES": 10000,
              "REPLAY_START_SIZE": 10000,
              "EPSILON_DECAY_LAST_FRAME": 10000,
              "NUMBER_FRAMES": 300000,
              "ACTION_SIZE": 10,  # network doesnt seem to care much about action_space discretization...
              "SKIP_NUMBER": 4,
              "EPSILON_START": 1,
              "EPSILON_FINAL": 0.02,
              "device": "cpu"}

# cpu faster than cuda, network is so small that the time needed to load it into the gpu is larger than
# the gained time of parallel computing


if __name__ == '__main__':

    # ______________________________PREPARE AGENT ENVIRONMENT, BUFFER, NETS _______________________________
    buffer, agent, net, tgt_net = helper_functions.setup_all(Parameters)

    helper_functions.load_buffer(agent, Parameters["REPLAY_START_SIZE"], Parameters["REPLAY_SIZE"])

    helper_functions.load_previous_model(net, tgt_net, "RoboschoolPong-v1-best.dat", Parameters["LOAD_PREVIOUS "])

    optimizer = optim.Adam(net.parameters(), lr=Parameters["LEARNING_RATE"])

    writer = SummaryWriter(comment="-" + "batch" + str(Parameters["BATCH_SIZE"]) + "_n" + str(agent.env.action_space.n) +
                                   "_eps" + str(Parameters["EPSILON_DECAY_LAST_FRAME"]) + "_skip" + str(4))

    if Parameters["LOAD_PREVIOUS "]:
        Parameters["EPSILON_START"] = Parameters["EPSILON_FINAL"]

    # ______________________________TRAINING__________________________________________________
    print("Start training: ")
    best_reward = - 1000  # Initialize at a very low value
    reward = []
    for frame in range(Parameters["NUMBER_FRAMES"]):
        epsilon = max(Parameters["EPSILON_FINAL"], Parameters["EPSILON_START"] - frame / Parameters["EPSILON_DECAY_LAST_FRAME"])
        ep_reward = agent.play_step(net, epsilon, Parameters["device"])
        if ep_reward:
            reward.append(ep_reward)
            writer.add_scalar("episode_reward", ep_reward, frame)
            writer.add_scalar("mean100_reward", np.mean(reward[-100:]), frame)

            if ep_reward > best_reward:
                best_reward = ep_reward
                writer.add_scalar("best reward", best_reward, frame)
                torch.save(net.state_dict(), Parameters["DEFAULT_ENV_NAME"] + "-best.dat")
        if (frame % Parameters["SYNC_TARGET_FRAMES"]) == 0:
            tgt_net.load_state_dict(net.state_dict())  # Syncs target and Standard net
            print("We are at: %i / %i frames" % (frame, Parameters["NUMBER_FRAMES"]))
            torch.save(net.state_dict(), Parameters["DEFAULT_ENV_NAME"] + "-time_update.dat")

        optimizer.zero_grad()
        batch = buffer.sample(Parameters["BATCH_SIZE"])
        loss_t = project_classes.calc_loss(batch, net, tgt_net, Parameters["GAMMA"], Parameters["device"])
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t, frame)
        writer.add_scalar("epsilon", epsilon, frame)

    writer.close()
    print("End training!")
    print("Time was: %f" % (time.time() - start_time))
    torch.save(net.state_dict(), Parameters["DEFAULT_ENV_NAME"] + "end_of_training.dat")

import gym
import roboschool

import roboschool.multiplayer

# game = roboschool.gym_pong.PongSceneMultiplayer()
# gameserver = roboschool.multiplayer.SharedMemoryServer(game, "Skynet", want_test_window=True)
# gameserver.serve_forever()

env = gym.make("RoboschoolPong-v1")

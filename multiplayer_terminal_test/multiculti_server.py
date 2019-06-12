import os, sys, subprocess
import numpy as np
import gym
import roboschool


if len(sys.argv)==1:
    import roboschool.multiplayer
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pongdemo", want_test_window=True)
    gameserver.serve_forever()

# Set Test window to false if you have video==True in the other script, else it wont work

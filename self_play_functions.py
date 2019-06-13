import roboschool.multiplayer


def start_server(session_name, render):
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, session_name, want_test_window=render)
    gameserver.serve_forever()
    return


def start_player2(session_name):
    pass
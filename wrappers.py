import gym
import collections
import numpy as np
import roboschool

'''
Contains the environment wrappers. The action_space_discretizer replaces the action_space of the environment
with gym.spaces.Discrete(n*n). It then overwrites the step() function so that integer action_numbers will be translated 
to float arrays [x1, x2] for the environment. This is necessary for Deep Q Learning. If we have an input n
the class will create n*n actions between [-1, -1] and [1, 1]
'''


class action_space_discretizer (gym.Wrapper):
    def __init__(self, env, n):
        super(action_space_discretizer, self).__init__(env)
        self.action_space = gym.spaces.discrete.Discrete(n*n)
        self.action_tuple = collections.deque(maxlen=n*n)
        x = np.linspace(-1, 1, n)
        for i in range(n):
            for j in range(n):
                self.action_tuple.append(np.array([x[i], x[j]]))

    def step(self, action):
        assert type(action) == int, "we discretized the actions! please give int between 0 and n**2"
        return self.env.step(self.action_tuple[int(action)])

    def reset(self):
        return self.env.reset()


# Skips a number of frames and repeats the last action for them. speeds up training extremely without
# sacrificing much accuracy
class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


#  Main is just for debugging
if __name__ == '__main__':
    env = gym.make("RoboschoolPong-v1")
    n = 2
    env = action_space_discretizer(env, n)
    for i in range(n*n):
        print(env.action_tuple[i])

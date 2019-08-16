import numpy as np
import matplotlib.pyplot as plt
import gym
import gym.wrappers

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback
import random

# Reference: https://mclearninglab.tistory.com/35

ENV_NAME = 'LunarLander-v2'

env = gym.make(ENV_NAME)
# To get repeatable results.
sd = 16
np.random.seed(sd)
random.seed(sd)
env.seed(sd)
nb_actions = env.action_space.n

#env = gym.wrappers.Monitor(env,'./monitor',force=True)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=500000, window_length=1)
policy = EpsGreedyQPolicy(eps=1.0)

class EpsDecayCallback(Callback):
    def __init__(self, eps_poilcy, decay_rate=0.95):
        self.eps_poilcy = eps_poilcy
        self.decay_rate = decay_rate
    def on_episode_begin(self, episode, logs={}):
        self.eps_poilcy.eps *= self.decay_rate
        print ('eps = %s' % self.eps_poilcy.eps)

class LivePlotCallback(Callback):
    def __init__(self, nb_episodes=4000, avgwindow=20):
        self.rewards = np.zeros(nb_episodes) - 1000.0
        self.X = np.arange(1, nb_episodes+1)
        self.avgrewards = np.zeros(nb_episodes) - 1000.0
        self.avgwindow = avgwindow
        self.rewardbuf = []
        self.episode = 0
        self.nb_episodes = nb_episodes
        plt.ion()
        self.fig = plt.figure()
        self.grphinst = plt.plot(self.X, self.rewards, color='b')[0]
        self.grphavg  = plt.plot(self.X, self.avgrewards, color='r')[0]
        plt.ylim([-450.0, 350.0])
        plt.xlabel('Episodes')
        plt.legend([self.grphinst, self.grphavg], ['Episode rewards', '20-episode-average-rewards'])
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='k', linestyle='--')

    def __del__(self):
        self.fig.savefig('monitor/plot.png')
        
    def on_episode_end(self, episode, logs):
        if self.episode >= self.nb_episodes:
            return
        rw = logs['episode_reward']
        self.rewardbuf.append(rw)
        if len(self.rewardbuf) > self.avgwindow:
            del self.rewardbuf[0]
        self.rewards[self.episode] = rw
        self.avgrewards[self.episode] = np.mean(self.rewardbuf)
        self.plot()
        self.episode += 1
    def plot(self):
        self.grphinst.set_ydata(self.rewards)
        self.grphavg.set_ydata(self.avgrewards)
        plt.draw()
        plt.pause(0.01)
        
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy, enable_double_dqn=False)
dqn.compile(Adam(lr=0.002, decay=2.25e-05), metrics=['mse'])

cbs = [EpsDecayCallback(eps_poilcy=policy, decay_rate=0.975)]
cbs += [LivePlotCallback(nb_episodes=4000, avgwindow=20)]
dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2, callbacks=cbs)

dqn.save_weights('monitor/dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# evaluate the algorithm for 100 episodes.
#dqn.test(env, nb_episodes=100, visualize=True)
import os
import numpy as np
import random
from collections import defaultdict
import gym
import environment

env = gym.make('maze-5x5-v0')

# Reference code: https://github.com/suhoy901/Reinforcement_Learning/blob/master/05.maze_sarsa/sarsa_basic.py

# State 의 boundary
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Maze의 size (10, 10)
NUM_GRID = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

class Agent:
    def __init__(self, actions):
        self.actions = actions
        self.discount_factor = 0.9 # 감가율
        self.epsilon = 0.1 # 엡실론
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        # TODO: 큐함수를 업데이트 하는 코드를 작성
        # self.discount_factor와 self.q_table을 이용하세요.

        # 구현을 완료했다면 아래 pass는 지우셔도 됩니다.
        pass

    # 입실론 탐욕 정책에 따라서 행동을 반환하는 메소드입니다.
    def get_action(self, state):
        # TODO: ε-탐욕 정책 코드를 작성
        # self.epsilon을 이용하세요.

        action = np.random.choice(self.actions)

        return int(action)

# 범위 밖으로 나간 state를 다시 maze안으로 넣어주는 코드
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_GRID[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_GRID[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_GRID[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    env.reset()
    agent = Agent(actions=list(range(env.action_space.n)))
    scores = []
    episodes = []

    for episode in range(250):
        state = env.reset()
        state = state_to_bucket(state)
        action = agent.get_action(state)
        total_reward = 0

        while True:
            env.render()

            next_state, reward, done, _ = env.step(action)
            next_state = state_to_bucket(next_state)
            next_action = agent.get_action(next_state)

            agent.learn(str(state), action, reward, str(next_state), next_action)
            total_reward += reward
            state = next_state
            action = next_action

            if done:
                print("Episode : %d total reward = %f . " % (episode, total_reward))
                episodes.append(episode)
                scores.append(total_reward)

                break
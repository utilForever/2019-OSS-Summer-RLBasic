import os
import numpy as np
import random
from collections import defaultdict
import gym
import environment
import time

env = gym.make('maze-5x5-v0')

# State 의 boundary
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# Maze의 size (10, 10)
NUM_GRID = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))

class Agent:
    def __init__(self, actions):
        self.actions = actions
        self.discount_factor = 0.9 # 감가율
        self.epsilon = 0.1 # 엡실론
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]) # 큐테이블

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + 0.2 *
                (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환하는 메소드입니다.
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[str(state)]
            action = self.arg_max(state_action)
        return int(action)
		
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

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
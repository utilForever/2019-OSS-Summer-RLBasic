import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

'''
일단 하이퍼파라미터에 None이라고 되어있는 부분 위주로 수정해주세요. (다른 것들 잘못 건드시면 안될수도 있음)
cartpole_dqn.py에 있는 예제 그대로 복사하셔도 됩니다.
하지만 이것 저것 수정해 보시면서 더 좋은 에이전트를 만들어 보는 것도 좋을 것 같습니다.
'''

# 최대로 실행할 에피소드 수를 설정합니다.
EPISODES = 2000

# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        '''
        구글 colab에서는 아래 render를 True로 만들면 실행이 안됩니다.
        '''
        self.render = False

        '''
        저장해 놓은 신경망 모델을 가져올 지 선택합니다. (lunarlander_trainded.h5)
        훈련을 중간에 중단해 놓았다가 다시 시작하려면 아래를 True로 바꾸고 실행하시면 됩니다.
        '''
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        '''
        일단 None이라고 되어있는 부분 위주로 수정해주세요. (다른 것들 잘못 건드시면 안될수도 있음)
        아래 8개 하이퍼파라미터(maxlen 포함)는 cartpole_dqn 예제 그대로 복사하셔도 되고, 좀 수정하셔도 됩니다.
        '''
        self.discount_factor = 0.99
        self.learning_rate = None
        self.epsilon = 1.0
        self.epsilon_decay = None
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 10000

        # 리플레이 메모리, 최대 크기 10000
        self.memory = deque(maxlen=20000)

        # 모델과 타깃 모델 생성
        '''
        아마 그냥 실행하시면 오류가 날텐데
        build_model을 완성하시면 오류가 사라집니다.
        '''
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("lunarlander_trainded.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        '''
        cartpole_dqn 파일의 예제를 그대로 사용하셔도 되고,
        좀 수정하셔도 됩니다.
        수정하신 뒤에는 아래에 있는 pass를 지워주세요.
        '''
        pass

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action_greedy(self, state):
        # 어제 큐함수 구현 과제 내용이니 넘어가겠습니다.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    # LunarLander-v2 환경, 최대 타임스텝 수가 500
    env = gym.make('LunarLander-v2')
    env.seed(0)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0

        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action_greedy(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)

            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                # 100 에피소드마다 학습 결과 
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("lunarlander_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                if e != 0 and e % 100 == 0:
                    print("Saved!")
                    agent.model.save_weights("lunarlander_trainded.h5")

                # 이전 50개 에피소드의 점수 평균이 200보다 크면 학습 중단
                if np.mean(scores[-min(50, len(scores)):]) > 200:
                    print("Success!")
                    agent.model.save_weights("lunarlander_trainded.h5")
                    sys.exit()

        # 무작위 행동으로 리플레이 메모리를 어느 정도 채운 다음부터
        # epsilon의 값을 epsilon_min 값까지 조금씩 줄여줍니다 
        if len(agent.memory) >= agent.train_start and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
import numpy as np
import random


class Agent:
    def __init__(self, learning_rate, discount_factor, epsilon, observation_space, nb_actions, env):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.eps = epsilon
        self.env = env
        self.q_table = np.zeros([observation_space, nb_actions])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        # update q-value with update rule
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def choose_action(self, state):
        if random.uniform(0, 1) < self.eps:
            # random action (exploration)
            action = self.env.action_space.sample()
        else:
            # greedy action (exploitation)
            action = np.argmax(self.q_table[state])
        return action

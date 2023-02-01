from customer import CustomerEnvironmentV2
import numpy as np
import random

'''
idea:

calculate mse for this update
if mse > threshold:
    reset everything

'''


class QLearningAgent:

    def __init__(self, env, eps_start=1.0, eps_decay=0.096, eps_min=0.001, gamma=0.99, alpha=0.99, change_detection_threshold=1e-5):
        self.env = env
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros(
            (env.n_actions, env.n_states), dtype=float)
        self.Q_old = np.zeros(
            (env.n_actions, env.n_states), dtype=float)
        self.Q_history = []
        self.reward_buffer = []
        self.mse_buffer = []
        self.prev_mse = 0.0
        self.iteration_cntr = 1

    def choose_action(self, state):
        '''
        Returns an e-greedy action
        '''
        epsilon = max(self.epsilon, self.eps_min)

        if random.uniform(0, 1) < epsilon:
            # Explore
            action = np.random.choice(range(self.env.n_actions))
            return action
        else:
            # Exploit
            actions = self.Q[:, state]

            return np.argmax(actions)

    def update_Q(self, memory):
        (state, action, state_, reward) = memory
        error = self.gamma*np.max(self.Q[:, state_]) - self.Q[action, state]

        '''
        Based on the Q-learning update function on https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
        '''
        self.Q[action, state] = self.Q[action, state] + \
            self.alpha * (reward + error)
        '''
        if self.iteration_cntr % 10 == 0:
            if self.iteration_cntr > 10:
                self.mse_buffer.append(self.calc_mse(self.Q, self.Q_old))
            self.Q_old = self.Q.copy()
        self.iteration_cntr += 1
        '''

    def save_Q_snapshot(self):
        self.Q_history.append(self.Q)

    def decay_eps(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.eps_decay*self.epsilon

    def reset_epsilon(self):
        self.epsilon = self.eps_start

    def reset_Q(self):
        self.Q = self.Q = np.zeros(
            (self.env.n_actions, self.env.n_states), dtype=float)

    def print_policy(self):
        policy = []
        for i in range(self.env.n_states):
            policy.append((i, np.argmax(self.Q[:, i])))
        print(policy)

    def build_policy(self):
        policy = []
        for i in range(self.env.n_states):
            policy.append(np.argmax(self.Q[:, i]))
        return policy

    def calc_mse(self, A, B):
        return (np.square(A - B)).mean(axis=None)


class RUQLAgent(QLearningAgent):
    def __init__(self, env, eps_start=1.0, eps_decay=0.096, eps_min=0.001, gamma=0.99, alpha=0.99, approximate=False):
        self.env = env
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros(
            (env.n_actions, env.n_states), dtype=float)
        self.Q_history = []
        self.approximate = approximate

    def update_Q(self, memory):
        # Altered update function to implement RUQL
        (state, action, state_, reward) = memory
        error = self.gamma*np.max(self.Q[:, state_]) - self.Q[action, state]
        pi_sa = np.max(self.Q[:, state])

        if self.approximate:
            if pi_sa != 0:
                a = (1.0-self.alpha)**(1/pi_sa)
                self.Q[action, state] = a*self.Q[action, state] + \
                    (1-a)*(reward + self.gamma*np.max(self.Q[:, state_]))
            else:
                self.Q[action, state] = self.Q[action, state] + \
                    self.alpha * (reward + error)
        else:
            if pi_sa != 0 and int(1/pi_sa) != 0:
                '''
                if int(1/pi_sa) > 1:
                    print(int(1/pi_sa))
                '''
                for _ in range(int(1/pi_sa)):
                    self.Q[action, state] = self.Q[action, state] + \
                        self.alpha * (reward + error)
            else:
                self.Q[action, state] = self.Q[action, state] + \
                    self.alpha * (reward + error)


class RUQLAgentV2(QLearningAgent):
    def __init__(self, env: CustomerEnvironmentV2, eps_start=1.0, eps_decay=0.096, eps_min=0.001, gamma=0.99, alpha=0.99):
        self.env = env
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros(
            (env.n_actions, env.n_days, env.n_hours), dtype=float)
        self.Q_history = []

    def update_Q(self, memory):
        # Altered update function to implement RUQL
        (state, action, state_, reward) = memory
        error = self.gamma * \
            np.max(self.Q[:, state_[0], state_[1]]) - \
            self.Q[action, state[0], state[1]]

        pi_sa = np.max(self.Q[:, state[0], state[1]])
        if pi_sa != 0 and int(1/pi_sa) != 0:
            '''
            if int(1/pi_sa) > 1:
                print(int(1/pi_sa))
            '''
            for _ in range(int(1/pi_sa)):
                self.Q[action, state[0], state[1]] = self.Q[action, state[0], state[1]] + \
                    self.alpha * (reward + error)
        else:
            self.Q[action, state[0], state[1]] = self.Q[action, state[0], state[1]] + \
                self.alpha * (reward + error)

    def choose_action(self, state):
        '''
        Returns an e-greedy action
        '''
        epsilon = max(self.epsilon, self.eps_min)

        if random.uniform(0, 1) < epsilon:
            # Explore
            action = np.random.choice(range(self.env.n_actions))
            return action
        else:
            # Exploit
            actions = self.Q[:, state[0], state[1]]

            return np.argmax(actions)

    def build_policy(self):
        policy = np.zeros(self.env.n_days, self.env.n_hours)
        for i in range(self.env.n_days):
            for j in range(self.env.n_hours):
                policy[i, j] = np.argmax(self.Q[:, i, j])
        return policy

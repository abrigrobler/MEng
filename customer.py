import numpy as np
import matplotlib.pyplot as plt
import random


class CustomerEnvironment:

    def __init__(self, n_actions, context_transition_probability, regenerate_rpfs_on_context_change=False):
        self.ctp = context_transition_probability
        self.n_actions = n_actions
        self.n_states = 24
        self.point_alter_probability = random.uniform(0.2, 0.8)
        self.response_probabilities = [
            self.generate_random_function() for _ in range(self.n_actions)]
        self.regenerate_rpfs_on_context_change = regenerate_rpfs_on_context_change

    def clip_rp_function(self, y, max_probability=0.9, epsilon=0.05):
        # Clip function
        for i in range(len(y)):
            if y[i] < epsilon:
                y[i] = epsilon

            if y[i] > max_probability:
                y[i] = max_probability
        return y

    def alter_function(self, y, severity, probability=0.2):
        for i in range(len(y)):
            if random.uniform(0, 1) < probability:
                if random.uniform(0, 1) > 0.5:
                    y[i] = y[i] + random.uniform(0, severity)
                else:
                    y[i] = y[i] - random.uniform(0, severity)
        return self.clip_rp_function(y)

    def change_context(self):
        if not self.regenerate_rpfs_on_context_change:
            if random.uniform(0, 1) < self.ctp:
                self.response_probabilities = [
                    self.alter_function(self.response_probabilities[i], severity=0.3, probability=self.point_alter_probability) for i in range(self.n_actions)]
                return True
            else:
                return False
        else:
            if random.uniform(0, 1) < self.ctp:
                self.response_probabilities = [
                    self.generate_random_function() for _ in range(self.n_actions)]
                return True
            else:
                return False

    def step(self, state, action):
        done = False

        p = random.uniform(0, 1)
        response_probability = self.response_probabilities[action][state]
        reward = 0
        if p < response_probability:
            reward = 1
        if state == 23:
            done = True
        else:
            state += 1

        return state, reward, done
    
    def test_single_time(self, state, action):
         p = random.uniform(0, 1)
         response_probability = self.response_probabilities[action][state]
         if p < response_probability:
             return True
         return False

    def generate_random_function(self, resolution=1.0, hours=24,
                                 horizontal_scale=15, N=10, max_probability=1.0, epsilon=0.05, L=0.4):
        t = np.arange(0, hours, 1.0/resolution)
        # Initialise function
        y = np.zeros(len(t), dtype=float)

        for i in range(N):
            c1 = random.uniform(-L, L)
            c2 = random.uniform(-L, L)
            # Construct function
            y += (c1*(np.sin((i+1)*t/horizontal_scale)) +
                  c2*(np.cos((i+1)*t/horizontal_scale)))

        # Clip function
        for i in range(len(y)):
            if y[i] < epsilon:
                y[i] = epsilon

            elif y[i] > max_probability:
                y[i] = max_probability

        return y

    def plot_response_probabilities(self):
        t = np.arange(0, 24, 1)
        fig, axs = plt.subplots(self.n_actions, sharex='col')
        fig.suptitle('Response probability functions for all actions')
        plt.xticks(t)
        for i in range(self.n_actions):
            axs[i].plot(self.response_probabilities[i])

        plt.show()


class CustomerEnvironmentV2(CustomerEnvironment):

    def __init__(self, n_actions, context_transition_probability):
        self.ctp = context_transition_probability
        self.n_actions = n_actions
        self.n_hours = 24
        self.n_days = 7
        self.response_probabilities = self.init_rpfs()

    def clip_rp_function(self, y, max_probability=0.9, epsilon=0.05):
        # Clip function
        for i in range(len(y)):
            if y[i] < epsilon:
                y[i] = epsilon

            elif y[i] > max_probability:
                y[i] = max_probability
        return y

    def alter_function(self, y, severity):
        for i in range(len(y)):
            if random.uniform(0, 1) > 0.5:
                y[i] = y[i] + severity
            else:
                y[i] = y[i] - severity
        return self.clip_rp_function(y)

    def init_rpfs(self):
        response_probabilities = []
        for _ in range(self.n_days):
            response_probabilities.append(
                [self.generate_random_function() for _ in range(self.n_actions)])
        return response_probabilities

    def change_context(self):
        if random.uniform(0, 1) < self.ctp:
            for d in self.response_probabilities:
                for i in range(self.num_actions):
                    d[i] = self.alter_function(d[i], 0.1)
            return True
        else:
            return False

    def step(self, state, action):
        done = False
        # Context transition

        p = random.uniform(0, 1)
        response_probability = self.response_probabilities[action][state[0]][state[1]]
        reward = 0
        if p < response_probability:
            reward = 1
        if state[0] == 6 and state[1] == 23:
            done = True
        elif state[1] == 23:
            state[0] += 1
            state[1] = 0
        else:
            state[1] += 1

        return state, reward, done

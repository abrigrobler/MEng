from customer import CustomerEnvironmentV2
from q_learning_agent import QLearningAgent, RUQLAgentV2
import random
import numpy as np
from console_progressbar import ProgressBar
import matplotlib.pyplot as plt

NUM_ACTIONS = 3


class Tester:

    def __init__(self):
        pass

    def calculate_agent_accuracy(self, dims, agent, response_probabilities):

        policy_optimal = np.zeros(dims, dtype=int)
        for i in range(dims[0]):
            for j in range(dims[1]):
                policy_optimal = np.argmax(response_probabilities[i][j])
        policy_agent = agent.build_policy()
        # print(policy_optimal)
        # print(policy_agent)
        correct = 0.0
        for i in range(len(policy_agent)):
            if policy_optimal[i] == policy_agent[i]:
                correct += 1
        return correct

    def run_single_trial(self, episodes=5000):
        env = CustomerEnvironmentV2(
            n_actions=NUM_ACTIONS, context_transition_probability=0.0008)

        agent_R = RUQLAgentV2(env=env, eps_decay=0.9996,
                              gamma=0.9, alpha=0.15)

        context_counter = 0
        accuracy_snapshots_R = []

        for i in range(episodes):
            state = 0

            while True:
                a_R = agent_R.choose_action(state)
                state_, reward_R, done = env.step(state, a_R)
                agent_R.update_Q((state, a_R, state_, reward_R))
                state = state_

                if done:
                    break
            old_response_probs = env.response_probabilities
            context_change_flag = env.change_context()
            if context_change_flag:
                correct_R = self.calculate_agent_accuracy(
                    agent_R, old_response_probs)
                accuracy_snapshots_R.append((correct_R*100/24))
                # print('Context transition, accuracy', (correct*100/24))
                # agent.reset_Q()
                # agent.reset_epsilon()
                context_counter += 1
                # pass
            # print('Episode', i, 'epsilon', agent.epsilon)
            # pb.print_progress_bar(i+1)

            agent_R.decay_eps()

        print("Agent_R chooses correctly %.2f percent of the time" %
              np.mean(accuracy_snapshots_R), 'with %d context transitions' % context_counter)

        # env.plot_response_probabilities()
        return np.mean(accuracy_snapshots_R)

    def run_full_comparison_trial(self):
        accuracy = []
        accuracy_R = []

        for i in range(20):
            print()
            print('For customer', i+1)
            a, a_R = self.run_single_trial()

            accuracy.append(a)
            accuracy_R.append(a_R)
        print()
        print('-----------------------------------------')
        print('TRIAL RESULTS')
        print('Q-Learning: %2.f accuracy' % np.mean(accuracy))
        print('RUCL: %2.f accuracy' % np.mean(accuracy_R))


# Tester.run_single_trial()
T = Tester()
T.run_single_trial()

from customer import CustomerEnvironment
from q_learning_agent import QLearningAgent, RUQLAgent
import random
import numpy as np
from console_progressbar import ProgressBar
import matplotlib.pyplot as plt
import csv

NUM_ACTIONS = 4


class Tester:

    def __init__(self):
        self.f = open('data.csv', 'w')
        self.writer = csv.writer(self.f)

    def calculate_agent_accuracy(self, agent, response_probabilities):
        policy_optimal = [np.argmax(np.array(response_probabilities)[
            :, i]) for i in range(24)]
        policy_agent = agent.build_policy()
        # print(policy_optimal)
        # print(policy_agent)
        correct = 0.0
        for i in range(len(policy_agent)):
            if policy_optimal[i] == policy_agent[i]:
                correct += 1
        return correct

    def run_single_trial(self, K, episodes=5000):
        step_counter = 0

        pb = ProgressBar(total=episodes)

        env = CustomerEnvironment(
            n_actions=NUM_ACTIONS, context_transition_probability=random.uniform(0.001, 0.1), regenerate_rpfs_on_context_change=False)

        agent = QLearningAgent(env=env, eps_decay=0.9996,
                               gamma=0.4, alpha=1e-4)

        agent_R = RUQLAgent(env=env, eps_decay=0.9996,
                            gamma=0.6, alpha=0.15, approximate=True)

        context_counter = 0
        accuracy_snapshots = []
        accuracy_snapshots_R = []

        for i in range(episodes):
            state = 0
            step_counter += 1

            while True:
                a = agent.choose_action(state)
                a_R = agent_R.choose_action(state)
                state_, reward, done = env.step(state, a)
                _, reward_R, _ = env.step(state, a_R)
                agent.update_Q((state, a, state_, reward))
                agent_R.update_Q((state, a_R, state_, reward_R))
                state = state_

                if done:
                    break
            old_response_probs = env.response_probabilities
            # env.plot_response_probabilities()

            context_change_flag = env.change_context()
            # env.plot_response_probabilities()
            if context_change_flag:
                correct = self.calculate_agent_accuracy(
                    agent, old_response_probs)
                correct_R = self.calculate_agent_accuracy(
                    agent_R, old_response_probs)
                accuracy_snapshots.append(
                    (step_counter/episodes)*(correct*100/24))
                accuracy_snapshots_R.append(
                    (step_counter/episodes)*(correct_R*100/24))

                # print('Context transition, accuracy', (correct*100/24))
                # agent.reset_Q()
                # agent.reset_epsilon()
                context_counter += 1
                step_counter = 0
                # pass
            # print('Episode', i, 'epsilon', agent.epsilon)
            # pb.print_progress_bar(i+1)

            agent.decay_eps()
            agent_R.decay_eps()

        agent_accuracy = 0.0
        agent_R_accuracy = 0.0
        if context_counter > 0:
            agent_accuracy = np.sum(accuracy_snapshots)
            agent_R_accuracy = np.sum(accuracy_snapshots_R)

        # plt.plot(agent.mse_buffer)
        # plt.show()

        print("Agent chooses correctly %.2f percent of the time" %
              agent_accuracy, 'with %d context transitions' % context_counter)
        print("Agent_R chooses correctly %.2f percent of the time" %
              agent_R_accuracy, 'with %d context transitions' % context_counter)
        self.writer.writerow(
            [K+1, agent_accuracy, agent_R_accuracy, context_counter])

        # env.plot_response_probabilities()
        return agent_accuracy, agent_R_accuracy

    def run_full_comparison_trial(self):
        accuracy = []
        accuracy_R = []

        for i in range(50):
            print()
            print('For customer', i+1)
            a, a_R = self.run_single_trial(i)
            accuracy.append(a)
            accuracy_R.append(a_R)
        print()
        print('-----------------------------------------')
        print('TRIAL RESULTS')
        print('Q-Learning: %2.f accuracy' % np.mean(accuracy))
        print('RUQL: %2.f accuracy' % np.mean(accuracy_R))
        self.f.close()


# Tester.run_single_trial()
T = Tester()
T.run_full_comparison_trial()

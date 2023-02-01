from customer import CustomerEnvironment
import numpy as np
import random
import matplotlib.pyplot as plt

NUM_CUSTOMERS = 10000
customers = []
responses_per_customer_mail = np.zeros(NUM_CUSTOMERS)
responses_per_customer_push = np.zeros(NUM_CUSTOMERS)
time_for_first_mailer=10-1
time_for_first_push=14-1

# Action 0 = email, action 1 = push notification

for i in range(NUM_CUSTOMERS):
    customers.append(CustomerEnvironment(n_actions=2, context_transition_probability=0.0))
    
for i in range(6*30):
    for idx in range(NUM_CUSTOMERS):
        if customers[idx].test_single_time(random.randint(time_for_first_mailer, 13), 0):
            responses_per_customer_mail[idx]+=1
        if customers[idx].test_single_time(random.randint(time_for_first_push, 19), 1):
            responses_per_customer_push[idx]+=1

responses_per_customer_mail=responses_per_customer_mail*100/(6*30)
responses_per_customer_push=responses_per_customer_push*100/(6*30)

print(np.mean(responses_per_customer_mail))
print(np.mean(responses_per_customer_push))
customers[0].plot_response_probabilities()
        
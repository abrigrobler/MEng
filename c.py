from customer import CustomerEnvironment

c = CustomerEnvironment(n_actions=4, context_transition_probability=0.0)

c.plot_response_probabilities()

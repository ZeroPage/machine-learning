import gym
import numpy as np
import random

epoch = 2000

def rargmax(vector):
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	a = random.choice(indices)
	return a

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
reward_list = []

for i in range(epoch):
	E = 1. / (i/100+1)
	dis = 0.9

	state = env.reset()
	total_reward = 0
	done = False

	while not done:
		# if np.random.rand(1) < E:
		# 	action = env.action_space.sample()
		# else:
		# 	action = np.argmax(Q[state, :])
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

		new_state, reward, done, _ = env.step(action)
		Q[state, action] = reward + dis * np.max(Q[new_state, :])

		state = new_state
		total_reward += reward
		i += 1

	reward_list.append(total_reward)

print(Q)
print("Success rate: " + str(sum(reward_list)/epoch))

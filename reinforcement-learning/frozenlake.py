import gym
import numpy as np
import random
from gym.envs.registration import register
import gym.envs.toy_text
import matplotlib.pyplot as plt

epoch = 2000
dis = 0.9
alpha = .085

register(
   	id='FrozenLake-v1',
   	entry_point='gym.envs.toy_text:FrozenLakeEnv',
   	kwargs={'map_name':'4x4', 'is_slippery':False}
)

env = gym.make('FrozenLake-v1')
Q = np.zeros([env.observation_space.n, env.action_space.n])
reward_list = []

def rargmax(vector):
	m = np.amax(vector)
	indices = np.nonzero(vector == m)[0]
	return random.choice(indices)

for i in range(epoch):
	state = env.reset()
	total_reward = 0
	done = False
	reward = 0

	while not done:
		action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) // (i+1))

		new_state, reward, done, _ = env.step(action)
		Q[state, action] = (1-alpha) * Q[state, action] + alpha * (reward + dis * np.max(Q[new_state, :]))

		state = new_state
		total_reward += reward

	reward_list.append(total_reward)

print(Q)
print("Success rate: " + str(sum(reward_list)/epoch * 100))
plt.bar(range(len(reward_list)), reward_list, color="blue")
plt.show()

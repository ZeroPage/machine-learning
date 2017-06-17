import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epoch = 2000
dis = 0.9
learing_rate=0.1

env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n
output_size = env.action_space.n

def onehot(s):
	return np.identity(16)[s:s+1]

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.1))

Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)
Qpred = tf.matmul(X, W)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)

rList = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epoch):
		state = env.reset()
		e = 1. / (i / 50 + 10)
		done = False
		rTotal = 0

		while not done:
			Qs = sess.run(Qpred, feed_dict={X: onehot(state)})
			if np.random.rand(1) < e:
				action = env.action_space.sample()
			else:
				action = np.argmax(Qs)

			new_state, reward, done, _ = env.step(action)
			if done:
				Qs[0, action] = reward
			else:
				Qs1 = sess.run(Qpred, feed_dict={X: onehot(new_state)})
				Qs[0, action] = reward + dis * np.max(Qs1)

			# Train after run
			sess.run(train, feed_dict={X: onehot(state), Y: Qs})
			state = new_state
			rTotal += reward
		rList.append(rTotal)

print("Success rate: " + str(sum(rList)/epoch * 100))
plt.bar(range(len(rList)), rList, 1/1.5, color="y")
plt.show()

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epoch = 1000
dis = 0.9
learing_rate=0.1

env = gym.make('CartPole-v0')

input_size = 4
output_size = env.action_space.n

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

W1 = tf.get_variable("W1", shape=[input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
B1 = tf.Variable(tf.random_uniform([128], 0, 0.01))
O1 = tf.nn.relu_layer(X, W1, B1)
O1 = tf.layers.dropout(O1, 0.8)

W2 = tf.Variable(tf.random_uniform([128, 256], 0, 0.1))
B2 = tf.Variable(tf.random_uniform([256], 0, 0.01))
O2 = tf.nn.relu_layer(O1, W2, B2)
O2 = tf.layers.dropout(O2, 0.8)

W3 = tf.Variable(tf.random_uniform([256, output_size], 0, 0.1))
B3 = tf.Variable(tf.random_uniform([output_size], 0, 0.01))
Qpred = tf.matmul(O2, W3) + B3

# For simple Q-network
# W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.1))
# Qpred = tf.matmul(X, W)

loss = tf.reduce_mean(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss)

rList = []
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epoch):
		state = env.reset()
		env.render()
		e = 1. / (i / 50 + 10)
		done = False
		step = 0

		while not done:
			Qs = sess.run(Qpred, feed_dict={X: np.reshape(state, (1, 4))})
			if np.random.rand(1) < e:
				action = env.action_space.sample()
			else:
				action = np.argmax(Qs)

			new_state, reward, done, _ = env.step(action)
			if done:
				Qs[0, action] = -100
			else:
				Qs1 = sess.run(Qpred, feed_dict={X: np.reshape(state, (1, 4))})
				Qs[0, action] = reward + dis * np.max(Qs1)

			# Train after run
			sess.run(train, feed_dict={X: np.reshape(state, (1, 4)), Y: Qs})
			state = new_state
			step += 1
		rList.append(step)

print("Success rate: " + str(sum(rList)/epoch))
plt.bar(range(len(rList)), rList, 1/1.5, color="y")
plt.show()

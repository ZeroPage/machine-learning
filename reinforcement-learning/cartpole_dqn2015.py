import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random

EPOCH = 5000
DIS = 0.9
L_RATE = 1e-1
MAX_QUEUE_SIZE = 50000

env = gym.make('CartPole-v0')

class DQN:
	def __init__(self, sess, input_n, output_n, name):
		self.sess = sess
		self.input_n = input_n
		self.output_n = output_n
		self.net_name = name

		self._build()

	def _build(self):
		with tf.variable_scope(self.net_name):
			self.X = tf.placeholder(shape=[None, self.input_n], dtype=tf.float32)

			W1 = tf.get_variable("W1", shape=[self.input_n, 17], initializer=tf.contrib.layers.xavier_initializer())
			B1 = tf.Variable(tf.random_uniform([17], 0, 0.01))
			L1 = tf.nn.relu_layer(self.X, W1, B1)
			# L1 = tf.layers.dropout(L1, 0.8)
			# L = tf.nn.tanh(tf.matmul(self.X, W1))

			W2 = tf.get_variable("W2", shape=[17, self.output_n], initializer=tf.contrib.layers.xavier_initializer())
			self.Qpred = tf.matmul(L1, W2)

		self.Y = tf.placeholder(shape=[None, self.output_n], dtype=tf.float32)
		self.loss = tf.reduce_mean(tf.square(self.Y - self.Qpred))
		self.train = tf.train.AdamOptimizer(learning_rate=L_RATE).minimize(self.loss)

	def predict(self, state):
		X_ = np.reshape(state, (1, self.input_n))
		return self.sess.run(self.Qpred, feed_dict={self.X: X_})

	def update(self, x_stack, y_stack):
		return self.sess.run([self.loss, self.train], feed_dict={self.X:x_stack, self.Y:y_stack})

def replay_train(trainDQN, subDQN, train_batch):
	x_stack = np.empty(0).reshape(0, trainDQN.input_n)
	y_stack = np.empty(0).reshape(0, trainDQN.output_n)

	# state, action, new_state, reward, done
	for state, action, new_state, reward, done in train_batch:
		Q = trainDQN.predict(state)

		if done:
			Q[0, action] = reward
		else:
			Q[0, action] = reward + DIS*np.max(subDQN.predict(new_state))

		x_stack = np.vstack([x_stack, state]) # stack state on x
		y_stack = np.vstack([y_stack, Q]) # stack Q on y
	return trainDQN.update(x_stack, y_stack)

def get_copy_vars(main_net="", sub_net=""):
	op_holder = []

	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_net)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=sub_net)
	for src_var, dest_var in zip(src_vars, dest_vars):
		# assign doesn't happens here
		op_holder.append(dest_var.assign(src_var.value()))

	# Return assign operators
	return op_holder

def bot_play(dqn):
	s = env.reset()
	total = 0
	while True:
		env.render()
		a = np.argmax(dqn.predict(s))
		s1, reward, done, _ = env.step(a)
		total += reward
		s = s1
		if done:
			print("Total score: {}".format(total))
			break

def main():
	replay_buffer = deque()

	with tf.Session() as sess:
		mainDQN = DQN(sess, env.observation_space.shape[0], env.action_space.n, "mainDQN")
		subDQN = DQN(sess, env.observation_space.shape[0], env.action_space.n, "subDQN")

		tf.global_variables_initializer().run()

		copy_dqn_ops = get_copy_vars(main_net=mainDQN.net_name, sub_net=subDQN.net_name)
		sess.run(copy_dqn_ops)

		for i in range(EPOCH):
			e = 1. / (i / 50 + 10)
			step = 0
			done = False
			state = env.reset()

			while not done:
				if np.random.rand(1) < e:
					action = env.action_space.sample()
				else:
					action = np.argmax(mainDQN.predict(state))

				new_state, reward, done, _ = env.step(action)
				if done:
					reward = -100

				replay_buffer.append((state, action, new_state, reward, done))
				if len(replay_buffer) > MAX_QUEUE_SIZE:
					replay_buffer.popleft()

				state = new_state
				step += 1

			if i % 10 == 1:
				for _ in range(50):
					minibatch = random.sample(replay_buffer, 10)
					replay_train(mainDQN, subDQN, minibatch)

				sess.run(copy_dqn_ops)

			# print("Total Step: {}".format(step))

		bot_play(mainDQN)

if __name__ == "__main__":
	main()
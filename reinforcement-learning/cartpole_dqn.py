import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random

epoch = 1000
dis = 0.9
l_rate = 0.1

env = gym.make('CartPole-v0')

class DQN:
	def __init__(self, sess, input_n, output_n, name):
		self.sess = sess
		self.input_n = input_n
		self.output_n = output_n
		self.net_name = name

		self.build()

	def build(self):
		with tf.variable_scope(self.net_name):
			self.X = tf.placeholder(shape=[None, self.input_n], dtype=tf.float32)

			W1 = tf.get_variable("W1", shape=[self.input_n, 16], initializer=tf.contrib.layers.xavier_initializer())
			L = tf.nn.tanh(tf.matmul(self.X, W1))

			W2 = tf.get_variable("W2", shape=[16, self.output_n], initializer=tf.contrib.layers.xavier_initializer())

			self.Qpred = tf.matmul(L, W2)

		self.Y = tf.placeholder(shape=[None, self.output_n], dtype=tf.float32)
		self.loss = tf.reduce_sum(tf.square(self.Y - self.Qpred))
		self.train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.loss)

	def predict(self, state):
		X_ = np.reshape(state, (1, self.input_n))
		return self.sess.run(self.Qpred, feed_dict={self.X: X_})

	def update(self, x_stack, y_stack):
		self.sess.run([self.loss, self.train], feed_dict={self.X:x_stack, self.Y:y_stack})
		return self.loss


MAX_QUEUE_SIZE = 50000

def replay_train(dqn, train_batch):
	x_stack = np.empty(0).reshape(0, dqn.input_n)
	y_stack = np.empty(0).reshape(0, dqn.output_n)

	for state, action, new_state, reward, done in train_batch:
		Q = dqn.predict(state)

		if done:
			Q[0, action] = reward
		else:
			Q[0, action] = reward + dis*np.max(dqn.predict(new_state))

		x_stack = np.vstack([x_stack, state]) # stack state on x
		y_stack = np.vstack([y_stack, Q]) # stack Q on y
	return dqn.update(x_stack, y_stack)

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
		dqn = DQN(sess, env.observation_space.shape[0], env.action_space.n, "mainDQN")
		tf.global_variables_initializer().run()

		for i in range(epoch):
			e = 1 / (i / 50 + 10)
			step = 0
			done = False
			state = env.reset()

			while not done:
				if np.random.rand(1) < e:
					action = env.action_space.sample()
				else:
					action = np.argmax(dqn.predict(state))

				new_state, reward, done, _ = env.step(action)

				if done:
					reward = -100

				replay_buffer.append((state, action, new_state, reward, done))
				if len(replay_buffer) > MAX_QUEUE_SIZE:
					replay_buffer.popleft()

				state = new_state
				step += 1
				if step > 10000:
					break

			if i % 10 == 1:
				for _ in range(50):
					minibatch = random.sample(replay_buffer, 10)
					loss = replay_train(dqn, minibatch)

			# print("Total Step: {}".format(step))

		bot_play(dqn)

if __name__ == "__main__":
	main()
import inspect
import os

import numpy as np
import tensorflow as tf
import time

class Vgg16:
    MEAN = {"blue": 103.939, "green": 116.779, "red": 123.68}
    __init = False
    __parameters = {}

    @classmethod
    def get_conv_filter(cls, name):
        with tf.name_scope(name) as scope:
            return tf.constant(cls.__data_dict[name][0], name='filter')

    @classmethod
    def get_bias(cls, name):
        with tf.name_scope(name) as scope:
            return tf.constant(cls.__data_dict[name][1], name='bias')

    @classmethod
    def get_fc_weight(cls, name):
        with tf.name_scope(name) as scope:
            return tf.constant(cls.__data_dict[name][0], name='weight')

    @classmethod
    def init(cls, vgg16_npy_path=None):
        if cls.__init:
            return None

        if vgg16_npy_path == None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path

        cls.__data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        for v in [
                "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2",
                "conv4_3", "conv5_1", "conv5_2", "conv5_3"
        ]:
            cls.__parameters[v] = {}
            cls.__parameters[v]["filter"] = cls.get_conv_filter(v)
            cls.__parameters[v]["bias"] = cls.get_bias(v)

        for v in range(6, 9):
            name = "fc" + str(v)
            cls.__parameters[name] = {}
            cls.__parameters[name]["weight"] = cls.get_fc_weight(name)
            cls.__parameters[name]["bias"] = cls.get_bias(name)

        cls.__init = True
        cls.__data_dict = None
        print("Load parameters")

    @classmethod
    def clear(cls):
        cls.__parameters = None
        cls.__data_dict = None
        cls.__init = False

    def __init__(self):
        if not self.__init:
            self.init()

    # images
    def build(self, rgb, normalization=True):
        if normalization:
            rgb_scale = rgb * 255
        else:
            rgb_scale = rgb

        # Convert RGB to BGR
        red, green, blue = tf.split(rgb_scale, 3, 3)
        red.shape.assert_is_compatible_with([None, 224, 224, 1])
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(
            [blue - self.MEAN["blue"], green - self.MEAN["green"], red - self.MEAN["red"]], 3)

        bgr.shape.assert_is_compatible_with([None, 224, 224, 3])

        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.__parameters[name]["filter"]
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.__parameters[name]["bias"]
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.__parameters[name]["weight"]
            biases = self.__parameters[name]["bias"]

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

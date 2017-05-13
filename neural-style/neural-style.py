import tensorflow as tf
import numpy as np
import scipy.misc as sm
import vgg19

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
BATCH_SIZE = 500


def read_image(path):
    return sm.imread(path).astype(np.float)

def save_image(image, path):
    image = np.clip(image, 0, 255).astype(np.uint8)
    sm.imsave(path, image)

def get_layer_style_loss(variable_layer, style_layer):
    shape = variable_layer.get_shape().as_list()
    N = shape[3]
    M = shape[1] * shape[2]
    reshaped_variable = tf.reshape(variable_layer, (-1, N))
    G = tf.matmul(reshaped_variable, reshaped_variable, transpose_a=True)
    reshaped_style = tf.reshape(style_layer, (-1, N))
    A = tf.matmul(reshaped_style, reshaped_style, transpose_a=True)
    return tf.nn.l2_loss(G - A) / (2.0 * N ** 2 * M ** 2)

contentImg = read_image("state.jpg")
styleImg   = read_image("style.jpg")
contentImg = tf.reshape(tf.image.resize_images(contentImg, [448, 448]), [-1, 448, 448, 3])
styleImg = tf.reshape(tf.image.resize_images(styleImg, [448, 448]), [-1, 448, 448, 3])

shape = contentImg.get_shape().as_list()
content_features = {}
style_features = {}

with tf.Session() as sess:
    img = tf.placeholder('float', shape=shape)
    net, mean_pixel = vgg19.net(VGG_PATH, img)

    ### calc vgg layers of content and style
    content_pre = vgg19.preprocess(contentImg, mean_pixel)
    content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(feed_dict={img: content_pre.eval()})
    style_pre = vgg19.preprocess(styleImg, mean_pixel)
    for layer in STYLE_LAYERS:
        style_features[layer] = net[layer].eval(feed_dict={img: style_pre.eval()})

    ### Image to be trained
    noise = np.random.normal(size=shape, scale=np.std(contentImg.eval()) * 0.1)
    init = content_pre * (1 - 0.0) + noise * 0.0
    image = tf.Variable(init, dtype='float')
    net, _ = vgg19.net(VGG_PATH, image)

### loss operators
content_loss = tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER])

style_loss = 0.0
for layer in STYLE_LAYERS:
    style_loss += get_layer_style_loss(net[layer], style_features[layer])
style_loss /= 5.0

ALPHA, BETA = 1.0 , 1e4
total_loss = ALPHA * content_loss + BETA * style_loss

### training operator
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(2e1, global_step, 10, 0.94, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(BATCH_SIZE):
        sess.run(train_op)
        print("{0}th loss: {1}".format(i, total_loss.eval()))
        if (i % 10 == 0):
            save_image(vgg19.unprocess(image.eval().reshape(shape[1:]), mean_pixel), 'output%03d.jpg' % i)
    save_image(vgg19.unprocess(image.eval().reshape(shape[1:]), mean_pixel), 'output.jpg')

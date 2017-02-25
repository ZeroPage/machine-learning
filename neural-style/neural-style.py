import tensorflow as tf
import numpy as np

import vgg16
# from vgg16 import Vgg16

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1

contents_queue = tf.train.string_input_producer(['cat.jpg', 'dog.jpg']) # List of contents image file
style_queue = tf.train.string_input_producer(['style.jpg']) # list of files to read

reader = tf.WholeFileReader()

filename, filecontents = reader.read(contents_queue)
content_images = tf.image.decode_jpeg(filecontents, channels=3) # use png or jpg decoder based on your files.

filename, filecontents = reader.read(style_queue)
style_images = tf.image.decode_jpeg(filecontents, channels=3)

resized_contents = tf.image.resize_images(content_images, [224,224]) / 255
resized_styles = tf.image.resize_images(style_images, [224,224]) / 255

min_after_dequeue = 100
batch_size = 1
capacity = min_after_dequeue + 3 * batch_size
content_batch, style_batch = tf.train.shuffle_batch(
    [resized_contents, resized_styles], batch_size=batch_size, capacity=capacity,
    min_after_dequeue=min_after_dequeue)

content_vgg = vgg16.Vgg16()
content_vgg.build(content_batch)
style_vgg = vgg16.Vgg16()
style_vgg.build(style_batch)
# _, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, _ = vgg.build(style_batch)

x = tf.Variable(tf.random_uniform([1, 224, 224, 3], minval=0, maxval=255, dtype=tf.float32))
variable_vgg = vgg16.Vgg16()
variable_vgg.build(x / 255)

result_image = tf.image.encode_png(tf.cast(x[0], tf.uint8))

content_loss = tf.nn.l2_loss(variable_vgg.conv4_2 - content_vgg.conv4_2)

def get_layer_style_loss(variable_layer, style_layer):
    shape = variable_layer.get_shape().as_list()
    N = shape[3]
    M = shape[1] * shape[2]
    reshaped_variable = tf.reshape(variable_layer, [-1, N])
    G = tf.matmul(reshaped_variable, reshaped_variable, transpose_a=True, transpose_b=False)
    reshaped_style = tf.reshape(style_layer, [-1, N])
    A = tf.matmul(reshaped_style, reshaped_style, transpose_a=True, transpose_b=False)
    return tf.nn.l2_loss(G - A) / (2 * N**2 * M ** 2)

E1_1 = get_layer_style_loss(variable_vgg.conv1_1, style_vgg.conv1_1)
E2_1 = get_layer_style_loss(variable_vgg.conv2_1, style_vgg.conv2_1)
E3_1 = get_layer_style_loss(variable_vgg.conv3_1, style_vgg.conv3_1)
E4_1 = get_layer_style_loss(variable_vgg.conv4_1, style_vgg.conv4_1)
E5_1 = get_layer_style_loss(variable_vgg.conv5_1, style_vgg.conv5_1)

style_loss = (E1_1 + E2_1 + E3_1 + E4_1 + E5_1) / 5

alpha, beta = 1, 1000
total_loss = alpha * content_loss + beta * style_loss

train_op = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(train_op)

    # for i in range(1): #length of your filename list
#    print(G1_1.get_shape())
#     for i in range(2):
        # content = resized_contents.eval() #here is your image Tensor :)
        # style = resized_styles.eval()

        # result = sess.run(content_vgg.prob)
        # print_prob(result[0], './vgg/synset.txt')

    # Image.imashow(Image.fromarray(np.asarray(content)))
    # pyplot.imshow(content)

    # matplotlib.pyplot.imshow(content)
    # matplotlib.pyplot.show()
    # print(sess.run(vgg.prob))
    # from vgg import utils
    # utils.print_prob(result[0], './vgg/synset.txt')

    coord.request_stop()
    coord.join(threads)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
from StringIO import StringIO

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from .vgg16 import Vgg16

FLAGS = None

def main(_):
    content_file = FLAGS.content_image
    style_file = FLAGS.style_image

    with tf.Graph().as_default() as g:
        Vgg16.init(StringIO(file_io.read_file_to_string(FLAGS.vgg16_npy))
)

        sess = tf.Session()

        content_data = tf.gfile.FastGFile(content_file, 'rb').read()
        content_image = None
        if content_file.endswith('png'):
            content_image = tf.image.decode_png(content_data, channels=3, name='content_image')
        elif content_file.endswith('jpg') or content_file.endswith('jpeg'):
            content_image = tf.image.decode_jpeg(content_data, channels=3, name='content_image')
        resized_content = tf.reshape(tf.image.resize_images(content_image, [224,224]) / 255, [-1, 224, 224, 3], name='resized_image')

        content_vgg = Vgg16()
        content_vgg.build(resized_content)

        style_data = tf.gfile.GFile(style_file, 'rb').read()
        style_image = None
        if style_file.endswith('png'):
            style_image = tf.image.decode_png(style_data, channels=3, name='style_image')
        elif style_file.endswith('jpg') or style_file.endswith('jpeg'):
            style_image = tf.image.decode_jpeg(style_data, channels=3, name='style_image')

        resized_style = tf.reshape(tf.image.resize_images(style_image, [224,224]) / 255, [-1, 224, 224, 3], name='resized_style')

        style_vgg = Vgg16()
        style_vgg.build(resized_style)


        # x = tf.Variable(tf.random_uniform([1, 224, 224, 3], minval=0, maxval=255)
        x = tf.Variable(resized_content, name='mixed_image')
        x_uint8 = tf.cast(x, tf.uint8)
        tf.summary.image('mix', x_uint8, max_outputs=FLAGS.max_steps // FLAGS.summary_step)

        variable_vgg = Vgg16()
        variable_vgg.build(x / 255)

        result_image = tf.image.encode_png(tf.cast(x[0], tf.uint8))
        write_image = tf.write_file(FLAGS.job_dir + '/output.png', result_image)

        content_loss = tf.nn.l2_loss(variable_vgg.conv4_2 - content_vgg.conv4_2, name='content_loss')
        tf.summary.scalar('content_loss', content_loss)

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

        style_loss = tf.divide((E1_1 + E2_1 + E3_1 + E4_1 + E5_1), 5.0, name='style_loss')
        tf.summary.scalar('style_loss', style_loss)

        alpha, beta = 1 / 1000, 1

        total_loss = tf.add(alpha * content_loss, beta * style_loss, name='total_loss')
        tf.summary.scalar('alpha_content_loss', alpha * content_loss)
        tf.summary.scalar('beta_style_loss', beta * style_loss)
        tf.summary.scalar('total_loss', total_loss)

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)

        init_op = tf.global_variables_initializer()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.job_dir, sess.graph)

        sess.run(init_op)

        for step in range(FLAGS.max_steps):
            if step % FLAGS.summary_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 10 == 0:
                print("step: ", step, ", loss :", total_loss.eval(session=sess))
            sess.run(train_op)

        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, FLAGS.max_steps)
        print("step:", FLAGS.max_steps, ", loss :", total_loss.eval(session=sess))

        summary_writer.flush()
        sess.run(write_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e+0,
        help='learning rate'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Number of steps to run trainer.'
    )

    parser.add_argument(
        '--summary-step',
        type=int,
        default=50,
        help='Number of steps to summary trainer.'
    )

    parser.add_argument(
        '--job-dir',
        type=str,
        default='job',
        help='Directory to put the model data.'
    )

    parser.add_argument(
        '--vgg16-npy',
        type=str,
        help="Filename to load vgg16 parameters"
    )

    parser.add_argument(
        '--content-image',
        type=str,
        required=True,
        help='filename for content image'
    )

    parser.add_argument(
        '--style-image',
        type=str,
        required=True,
        help='filename for style image'
    )

    parser.add_argument(
        '--verbose-logging',
        type=bool,
        default=False,
        help='Switch to turn on or off verbose logging and warnings'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

if sys.version_info > (3,0):
    from io import StringIO
else:
    from StringIO import StringIO

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python import debug as tf_debug

from .vgg16 import Vgg16

FLAGS = None

def main(_):
    style_file = FLAGS.style_image

    with tf.Graph().as_default() as g:
        if sys.version_info > (3,0):
            Vgg16.init(FLAGS.vgg16_npy)
        else:
            Vgg16.init(StringIO(file_io.read_file_to_string(FLAGS.vgg16_npy)))

        # Config to turn on JIT compilation
        config = tf.ConfigProto()
        if FLAGS.xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        sess = tf.Session(config=config)

        # if FLAGS.debug:
        #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #     sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        style_data = tf.gfile.GFile(style_file, 'rb').read()
        style_image = None
        if style_file.endswith('png'):
            style_image = tf.image.decode_png(style_data, channels=3, name='style_image')
        elif style_file.endswith('jpg') or style_file.endswith('jpeg'):
            style_image = tf.image.decode_jpeg(style_data, channels=3, name='style_image')

        resized_style = tf.reshape(tf.image.resize_images(style_image, [224,224]) / 255.0, [-1, 224, 224, 3], name='resized_style')

        style_vgg = Vgg16()
        style_vgg.build(resized_style)

        x = tf.Variable(tf.cast(tf.random_uniform([1, 224, 224, 3], minval=0, maxval=256, dtype=tf.int32), tf.float32), name='texture_image', dtype=tf.float32)
        x_uint8 = tf.cast(x, tf.uint8)
        tf.summary.image('texture', x_uint8, max_outputs=FLAGS.max_steps // FLAGS.summary_step)

        variable_vgg = Vgg16()
        variable_vgg.build(x / 255.0)

        result_image = tf.image.encode_png(tf.cast(x[0], tf.uint8))
        write_image = tf.write_file(FLAGS.job_dir + '/style.png', result_image)

        def get_style_error(style, x, name=None):
            with tf.variable_scope(name):
                batch, i, j, n = style.shape.as_list()
                m = i * j
                assert batch == 1
                print(m)
                reshape_style = tf.reshape(style, [-1, n])
                reshape_x = tf.reshape(x, [-1, n])

                gram_style = tf.matmul(reshape_style, reshape_style, transpose_a=True, transpose_b=False)
                print(gram_style.shape)
                tf.summary.tensor_summary("gram_style", gram_style)

                gram_x = tf.matmul(reshape_x, reshape_x, transpose_a=True, transpose_b=False)
                print(gram_style.shape)
                tf.summary.tensor_summary("gram_matrix", gram_x)

                return tf.nn.l2_loss(gram_style - gram_x) / (2 * (m ** 2) * (n ** 2))

        style_loss = get_style_error(style_vgg.conv1_1, variable_vgg.conv1_1, "conv1_1")
        tf.summary.scalar('style_loss', style_loss)

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(style_loss)

        init_op = tf.global_variables_initializer()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.job_dir, sess.graph)

        sess.run(init_op)

        for step in range(FLAGS.max_steps):
            if step % FLAGS.summary_step == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 10 == 0:
                print("step: ", step, ", loss :", style_loss.eval(session=sess))
            sess.run(train_op)

        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, FLAGS.max_steps)
        print("step:", FLAGS.max_steps, ", loss :", style_loss.eval(session=sess))

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
        '--style-image',
        type=str,
        required=True,
        help='filename for style image'
    )

    parser.add_argument(
        '--vgg16-npy',
        type=str,
        help="Filename to load vgg16 parameters"
    )

    parser.add_argument(
        '--verbose-logging',
        type=bool,
        default=False,
        help='Switch to turn on or off verbose logging and warnings'
    )

    parser.add_argument(
        '--xla',
        type=bool,
        default=False,
        help='Switch to turn on or off xla support'
    )

    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='Turn on debug CLI'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

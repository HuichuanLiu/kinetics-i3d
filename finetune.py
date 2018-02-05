import tensorflow as tf
from data_preprocess import *
import i3d
import tensorflow.contrib.slim as slim
import tensorboard

# ---------------------------set training------------------------------#
tf.app.flags.DEFINE_bool('is_training', True,
                         'if is training')

tf.app.flags.DEFINE_integer('max_epoch', 50,
                            'max epoch in training')

tf.app.flags.DEFINE_integer('max_step', 250,
                            'maximum steps in 1 epoch')

tf.app.flags.DEFINE_float('momentum', 0.9,
                          'momentum value for momentum optimizer')

tf.app.flags.DEFINE_float('i3d_lr', 1e-5,
                          'learning rate for pre-trained i3d payers')

tf.app.flags.DEFINE_float('logits_lr', 1e-3,
                          'learning rate for final logit layer')

tf.app.flags.DEFINE_string('train_log_dir', './data/train_log/',
                           'directory to save trianing logs')

tf.app.flags.DEFINE_string('eval_log_dir', './data/eval_log',
                           'directory to save evaluation logs')

tf.app.flags.DEFINE_integer('log_frequency', 5,
                            'frequency of logging by steps')

tf.app.flags.DEFINE_integer('eval_frequency', 10000000000,
                            'frequency of evaluation by steps')

tf.app.flags.DEFINE_string('ckpt_dir', './data/checkpoints/fine_tune_ckpt/model0.ckpt',
                           'directory to save checkpoint files')

tf.app.flags.DEFINE_integer('ckpt_frq', 1,
                            'frequency of saving checkpoint by epoch')

tf.app.flags.DEFINE_string('pre_train_ckpt', './data/checkpoints/flow_imagenet/model.ckpt',
                           'path to pre-trained checkpoints')

tf.app.flags.DEFINE_float('i3d_weight_decay', 1e-7,
                          'weight decay for i3d layers')

tf.app.flags.DEFINE_float('logits_weight_decay', 1e-7,
                          'weight decay for i3d layers')



# -----------------------------set data--------------------------------#
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'batch size in training')

tf.app.flags.DEFINE_integer('frame_num', 200,
                            'frame_num in each sample')

tf.app.flags.DEFINE_string('train_data', './data/train',
                           'directory to save train_samples')

tf.app.flags.DEFINE_string('eval_data', './data/eval',
                           'directory to save eval_samples')

tf.app.flags.DEFINE_integer('num_class', 1,
                            'number of classes in labels')
# -----------------------------set finished ---------------------------#
FLAGS = tf.flags.FLAGS


def run_model():
    global_step = tf.constant(0, dtype=tf.int64)
    # Flow input has only 2 channels.
    input_flow = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.frame_num, 224, 224, 2))
    input_labels = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.num_class))

    # ----------------------restore and edit graph--------------------------#
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.variable_scope('Flow'):
        # resotre flow model
        flow_model = i3d.InceptionI3d(FLAGS.num_class, spatial_squeeze=True, final_endpoint='Logits')

        if FLAGS.is_training:
            flow_logits, _ = flow_model(input_flow, is_training=True, dropout_keep_prob=0.5)
            loss = slim.losses.sigmoid_cross_entropy(flow_logits, input_labels)

            predictions = tf.round(tf.nn.sigmoid(flow_logits))
            correct_prediction = tf.equal(predictions, input_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            recall, update_recall = tf.metrics.recall(predictions=predictions, labels=input_labels)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('recall', recall)

            var_to_restore = slim.get_variables_to_restore(exclude=['Flow/inception_i3d/Logits'])
            var_to_retrain = slim.get_variables_to_restore(include=['Flow/inception_i3d/Logits'])
            flow_saver = tf.train.Saver(var_list=var_to_restore, reshape=True)

            global_step = tf.Variable(0)
            i3d_lr = tf.train.exponential_decay(FLAGS.i3d_lr, global_step=global_step, decay_steps=FLAGS.max_step, decay_rate=FLAGS.i3d_weight_decay)
            logit_lr = tf.train.exponential_decay(FLAGS.logits_lr, global_step=global_step, decay_steps=FLAGS.max_step, decay_rate=FLAGS.logits_weight_decay)
            train_i3d = slim.train.MomentumOptimizer(i3d_lr, FLAGS.momentum).\
                minimize(loss, var_list=var_to_restore, global_step = global_step)

            train_logits = slim.train.MomentumOptimizer(logit_lr, FLAGS.momentum).\
                minimize(loss, var_list=var_to_retrain, global_step = global_step)
            train_op = tf.group(train_i3d, train_logits)

            data_dir = FLAGS.train_data
        else:
            flow_logits, _ = flow_model(input_flow, is_training=False, dropout_keep_prob=1)
            var_to_restore = slim.get_variables_to_restore(exclude=['RGB'])
            flow_saver = tf.train.Saver(var_list=var_to_restore, reshape=True)

            predictions = tf.round(tf.nn.sigmoid(flow_logits))
            correct_prediction = tf.equal(predictions, input_labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            recall, update_recall = slim.metrics.streaming_recall(predictions=predictions, labels=input_labels)
            precision, update_precision = slim.metrics.streaming_precision(labels=input_labels, predictions=predictions)
            tpr, update_tpr = slim.metrics.streaming_true_positives(labels=input_labels, predictions=predictions)
            # conf_mat, update_cm = slim.metrics.confusion_matrix(labels=input_labels, predictions=predictions)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('recall', recall)
            tf.summary.scalar('precision', precision)
            tf.summary.scalar('tpr', tpr)
            # tf.summary.scalar('confused_matrix', conf_mat)

            up_date_metrics = tf.group(update_recall, update_recall, update_precision)

            data_dir = FLAGS.eval_data

    # -------------------------data set preparing---------------------------#

    filenames, labels = _get_files(data_dir)
    FLAGS.max_step = min(len(labels) // FLAGS.batch_size, FLAGS.max_step)
    FLAGS.max_step = (FLAGS.max_step - FLAGS.max_step % FLAGS.batch_size) / FLAGS.batch_size
    filenames = filenames[0:FLAGS.max_step * FLAGS.batch_size]
    labels = labels[0:FLAGS.max_step * FLAGS.batch_size]

    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda filename, label:
                          tuple(tf.py_func(_calc_opt_flow, [filename, label], [tf.float64, label.dtype])))
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()

    var_init = tf.group(tf.initialize_local_variables(), tf.initialize_all_variables())
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(var_init)

        if FLAGS.is_training:
            flow_saver.restore(sess, FLAGS.pre_train_ckpt)
            writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
        else:
            flow_saver.restore(sess, FLAGS.ckpt_dir)
            writer = tf.summary.FileWriter(FLAGS.eval_log_dir, sess.graph)

        next_batch = iterator.get_next()

        if not FLAGS.is_training:
            FLAGS.epoch = 1

        for epoch in range(FLAGS.max_epoch):
            sess.run(iterator.initializer)
            for step in range(FLAGS.max_step):
                print(step)
                optical_flow, label = sess.run(next_batch)

                if FLAGS.is_training:
                    sess.run(train_op,
                             feed_dict={input_flow: optical_flow, input_labels: label})
                else:
                    sess.run([accuracy, up_date_metrics],
                             feed_dict={input_flow: optical_flow, input_labels: label})

                if step % FLAGS.log_frequency == 0:
                    rs = sess.run(merged,
                                  feed_dict={input_flow: optical_flow, input_labels: label})
                    writer.add_summary(rs, step)

            if FLAGS.is_training and epoch % FLAGS.ckpt_frq == 0:
                var_to_store = slim.get_variables_to_restore(exclude=[])
                flow_saver = tf.train.Saver(var_list=var_to_store, reshape=True)
                flow_saver.save(sess, './data/checkpoints/fine_tune_ckpt/model%d.ckpt' % epoch)


def _calc_opt_flow(filename, label, frame_num=FLAGS.frame_num):
    optical_flow = get_optical_flow(filename, 224, frame_num)
    return optical_flow, label


def _get_files(dir_path):
    if not os.path.exists(dir_path):
        raise IOError

    images = []
    labels = []
    for root, dirs, files in os.walk(dir_path + '/0'):
        for file in files:
            images.append("%s/0/%s" % (dir_path, file))
            labels.append([0.])

    for root, dirs, files in os.walk(dir_path + '/1'):
        for file in files:
            images.append("%s/1/%s" % (dir_path, file))
            labels.append([1.])

    return images, labels


run_model()
# if __name__ == '__main__':
#     tf.app.run(run_model)

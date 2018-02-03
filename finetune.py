import tensorflow as tf
from data_preprocess import *
import i3d
import tensorflow.contrib.slim as slim

# ---------------------------set training------------------------------#
tf.app.flags.DEFINE_integer('max_epoch', 100,
                            'max epoch in training')

tf.app.flags.DEFINE_integer('max_step', 1000,
                            'maximum steps in 1 epoch')

tf.app.flags.DEFINE_float('momentum', 0.9,
                          'momentum value for momentum optimizer')

tf.app.flags.DEFINE_float('i3d_lr', 1e5,
                          'learning rate for pre-trained i3d payers')

tf.app.flags.DEFINE_float('lgt', 1e-3,
                          'learning rate for final logit layer')

tf.app.flags.DEFINE_string('train_log_dir', './data/train_log/',
                           'directory to save trianing logs')

tf.app.flags.DEFINE_string('eval_log_dir', './data/eval_log',
                           'directory to save evaluation logs')

tf.app.flags.DEFINE_integer('log_frequency', 10,
                            'frequency of logging by steps')

tf.app.flags.DEFINE_string('ckpt_dir', './checkpoints',
                           'directory to save checkpoint files')

tf.app.flags.DEFINE_integer('ckpt_frq', 1,
                            'frequency of saving checkpoint')

tf.app.flags.DEFINE_string('pre_train_ckpt', './data/checkpoint/flow_imagenet/model.ckpt',
                           'path to pre-trained checkpoints')

# -----------------------------set data--------------------------------#
tf.app.flags.DEFINE_integer('batch_size', 16,
                            'batch size in training')

tf.app.flags.DEFINE_integer('frame_num', 200,
                            'frame_num in each sample')

tf.app.flags.DEFINE_string('train_data', './data/train_samples',
                           'directory to save train_samples')

tf.app.flags.DEFINE_integer('num_class', 2,
                            'number of classes in labels')
#-----------------------------set finished ---------------------------#
FLAGS = tf.app.flags

def train():
    global_step = 0
    # Flow input has only 2 channels.
    input_flow = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, FLAGS.frame_num, 224, 224, 2))
    input_labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.num_class))

    #----------------------restore and edit graph--------------------------#
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.variable_scope('Flow'):
        # resotre flow model
        flow_model = i3d.InceptionI3d(FLAGS.num_class, spatial_squeeze=True, final_endpoint='Logits')
        flow_logits, _ = flow_model(input_flow, is_training=True, dropout_keep_prob=0.5)

        loss = slim.losses.sigmoid_cross_entropy(flow_logits, input_labels)


    #-------------------------redefine training----------------------------#
    var_to_restore = slim.get_variables_to_restore(exclude=["Logits,RGB"])
    var_to_retrain=[]
    flow_saver = tf.train.Saver(var_list=var_to_restore, reshape=True)

    train_i3d = slim.train.MomentumOptimizer(FLAGS.i3d_lr,FLAGS.momentum).minimize(flow_logits, var_list=var_to_restore, global_step=global_step)
    train_logits = slim.train.MomentumOptimizer(FLAGS.logit_lr, FLAGS.momentum).minimize(flow_logits, var_list=var_to_retrain, global_step = global_step)

    train_op = tf.group(train_i3d,train_logits)

    #-------------------------metrics in training--------------------------#

    predictions = tf.nn.sigmoid(flow_logits)
    accuracy = slim.metrics.accuracy(predictions,input_labels)

    optical_flow = []
    labels = []

    #-------------------------data set preparing---------------------------#

    filenames, labels = _get_files('./data/train')

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _calc_opt_flow, [filename, label], [tf.float64, label.dtype])))
    dataset = dataset.map(_calc_opt_flow)
    dataset = dataset.batch(16)
    iterator = dataset.make_initializable_iterator()
    var_init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(var_init)
        flow_saver.restore(sess, FLAGS.pre_train_ckpt)
        sess.run(iterator.initializer)

        for epoch in range(FLAGS.max_epoch):
            for step in range(FLAGS.max_step):
                sess.run(train_op, feed_dict={input_flow:optical_flow,input_labels:labels})
                if step%FLAGS.log_seq == 0:
                    pass

            if epoch%FLAGS.eval_seq == 0:
                evaluation()


def evaluation():
    pass

def _calc_opt_flow(filename, label):
    optical_flow = get_optical_flow(filename)
    return optical_flow, label

def _get_files(dir_path):
    if not os.path.exists(dir_path):
        raise IOError

    images = []
    labels = []
    for root, dirs, files in os.walk(dir_path+'/0'):
        for file in files:
            images.append("%s/0/%s" % (dir_path,file))
            labels.append([0])

    for root, dirs, files in os.walk(dir_path+'/1'):
        for file in files:
            images.append("%s/1/%s" % (dir_path,file))
            labels.append([1])

    return images, labels

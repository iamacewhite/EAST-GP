import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')
tf.app.flags.DEFINE_integer('num_classes', 96, '')
tf.app.flags.DEFINE_integer('hidden_size', 128, '')
tf.app.flags.DEFINE_string('base_model', 'resnet_v1_101', '')
tf.app.flags.DEFINE_string('loss', 'dice', '')

from nets import resnet_v1, resnet_v2
from nets import inception_v4
from nets import inception_resnet_v2
from tensorflow.python.ops import array_ops
#import pdb
FLAGS = tf.app.flags.FLAGS

def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def model(images, valid_affines, seq_len, mask, weight_decay=1e-5, is_training=True, model=FLAGS.base_model):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images, [128,128,128])
    if model == "reset_v1_50":
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')
        features = ['pool5', 'pool4', 'pool3', 'pool2']
    elif model == "resnet_v1_101":
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_101(images, is_training=is_training, scope='resnet_v1_101')
        features = ['pool5', 'pool4', 'pool3', 'pool2']
    elif model == "resnet_v2_101":
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet_v2.resnet_v2_101(images, is_training=is_training, scope='resnet_v2_101')
        features = ['pool5', 'pool4', 'pool3', 'pool2']
    elif model == "inception_v4":
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits, end_points = inception_v4.inception_v4(images, num_classes=None, is_training=is_training, scope='inception_v4')
        features = ['Mixed_7b', 'Mixed_6b' 'Mixed5a', 'Mixed_3a']
    elif model == "inception_resnet_v2":
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(images, num_classes=None, is_training=is_training, scope='inception_resnet_v2')
        features = ['Mixed_7a', 'Mixed_6a', 'Mixed_5b', 'MaxPool_3a_3x3']
    #pretty(end_points)
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            f = [end_points[fea] for fea in features]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    c1_2 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    a = slim.conv2d(slim.conv2d(c1_1, num_outputs[i], 3), num_outputs[i] // 2, 3)
                    b = slim.conv2d(c1_2, num_outputs[i] // 2, 3)
                    h[i] = tf.concat([a, b], axis=-1)
                    #h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    # g[i] = slim.conv2d(slim.conv2d(h[i], num_outputs[i], 3), num_outputs[i], 3)
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

            text_proposals = roi_rotate(g[3], valid_affines, mask)
            print('Shape after ROI rotate: {}'.format(text_proposals.shape))
            logits = lstm_ctc(text_proposals, seq_len)

            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            # 4 channel of axis aligned bbox and 1 channel rotation angle
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry, logits, text_proposals


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    # prediction_tensor = prediction_tensor * training_mask
    # target_tensor = target_tensor * training_mask
    sigmoid_p = prediction_tensor
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                                                            - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    loss = tf.reduce_mean(per_entry_cross_ent)
    loss *= 0.12
    focal_sum = tf.summary.scalar('focal_loss', loss)
    return loss, focal_sum


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    loss *= 0.01
    dice_sum = tf.summary.scalar('classification_dice_loss', loss)
    return loss, dice_sum

def roi_rotate(shared_feature, affine, mask):
    affine = tf.reshape(affine, [-1, 9])[:, :8]
    transformed = tf.contrib.image.transform(shared_feature, affine, interpolation='BILINEAR')
    masked_feature = (transformed * mask)[:, :8, :, :]
    return masked_feature

def lstm_cell():
    return tf.contrib.rnn.LSTMCell(FLAGS.hidden_size, state_is_tuple=True)

def lstm_ctc(features, seq_len):
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    # inputs, features = convolutional_layers()
    # print features.get_shape()

    # inputs = tf.placeholder(tf.float32, [None, None, common.OUTPUT_SHAPE[0]])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.


    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    print('Shape in ctc: {}'.format(features.shape))
    features = tf.transpose(features, perm=[0, 2, 1, 3])
    features = tf.reshape(features, [-1, features.shape[1], features.shape[2]* features.shape[3]])
    print('Shape in ctc: {}'.format(features.shape))

    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(0, 2)],
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, features, seq_len, dtype=tf.float32)

    shape = tf.shape(features)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, FLAGS.hidden_size])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([FLAGS.hidden_size,
                                         FLAGS.num_classes],
                                        stddev=0.1), name="W")
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[FLAGS.num_classes]), name="b")

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, FLAGS.num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    return logits

def ctc_loss(targets, logits, seq_len):
    loss = tf.nn.ctc_loss(targets, logits, seq_len, time_major=True, ignore_longer_outputs_than_inputs=True)
    cost = tf.reduce_mean(loss)
    ctc_sum = tf.summary.scalar('ctc_loss', cost)
    return cost, ctc_sum

def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask, targets, logits, seq_len):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    if FLAGS.loss == 'dice':
        classification_loss, dice_sum = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    #classification_loss, focal_sum = focal_loss(y_pred_cls, y_true_cls)
        #classification_loss *= 0.01
    else:
        #pdb.set_trace()
    # scale classification loss to match the iou loss part
        classification_loss, dice_sum = focal_loss(tf.reshape(y_pred_cls*(1-training_mask), [-1, FLAGS.text_scale*FLAGS.text_scale, 1]), tf.reshape(y_true_cls*(1-training_mask), [-1, FLAGS.text_scale*FLAGS.text_scale, 1]))
        #classification_loss *= 0.02
    recon_loss, ctc_sum = ctc_loss(targets, logits, seq_len)

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    aabb_sum = tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    theta_sum = tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss + recon_loss, [aabb_sum, theta_sum, dice_sum, ctc_sum]

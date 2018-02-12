import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('input_size', 576, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 16, '')
tf.app.flags.DEFINE_integer('num_readers', 12, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 400000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

import model
import icdar
import cv2
FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))
print "checkpoint path {}".format(FLAGS.checkpoint_path)

def tower_loss(images, score_maps, geo_maps, training_masks, valid_affines, seq_len, mask, targets, reuse_variables=None, is_training=True):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry, logits, text_proposals, before_rotate = model.model(images, valid_affines, seq_len, mask, is_training=is_training)

    model_loss, loss_sum_list = model.loss(score_maps, f_score, geo_maps, f_geometry, training_masks, targets, logits, seq_len)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        input_sum = tf.summary.image('input/input', images)
        score_map_sum = tf.summary.image('score_map', score_maps)
        score_map_pred_sum = tf.summary.image('score_map_pred', f_score * 255)
        geo_map_sum = tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        geo_map_pred_sum = tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        detection_mask_sum = tf.summary.image('training_masks', training_masks)
        recongnition_mask_sum = tf.summary.image('input/feature_mask', mask)
        model_loss_sum = tf.summary.scalar('model_loss', model_loss)
        total_loss_sum = tf.summary.scalar('total_loss', total_loss)
        model_val_sum = tf.summary.scalar('model_val_loss', model_loss)
        total_val_sum = tf.summary.scalar('total_val_loss', total_loss)
        rotate_feature_sum = tf.summary.image('input/rotated_feature', text_proposals[:, :, :, 0:1])
        before_rotate_feature_sum = tf.summary.image('input/before_rotated_feature', before_rotate[:, :, :, 0:1])

    return total_loss, model_loss, [model_loss_sum, total_loss_sum, input_sum, score_map_sum, score_map_pred_sum, geo_map_sum, geo_map_pred_sum, detection_mask_sum, recongnition_mask_sum, rotate_feature_sum, before_rotate_feature_sum] + loss_sum_list, [model_val_sum, total_val_sum]


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    # valid_bboxes = tf.placeholder(tf.float32, shape=[None, 4, 2])
    valid_labels = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])
    valid_affines = tf.placeholder(tf.float32, shape=[None, 8])
    text_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='text_masks')

    # model_val_loss = tf.placeholder(tf.float32, shape=[], name='model_val_loss')
    # total_val_loss = tf.placeholder(tf.float32, shape=[], name='total_val_loss')
    is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=2000, decay_rate=0.94, staircase=True)
    # add summary
    lr_sum = tf.summary.scalar('learning_rate', learning_rate)
    # val_model_sum = tf.summary.scalar('model_val_loss', model_val_loss)
    # val_total_sum = tf.summary.scalar('total_val_loss', total_val_loss)
    opt = tf.train.AdamOptimizer(learning_rate)
    transform_op = tf.contrib.image.transform(input_images, valid_affines, interpolation="BILINEAR")
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)


    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    train_sums = []
    test_sums = []

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    #if FLAGS.pretrained_model_path is not None:
    #    variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, tf.get_collection(tf.GraphKeys.MODEL_VARIABLES),
    #                                                        ignore_missing_vars=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(init)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        val_data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus),
                                         val=True)

        
        data = next(data_generator)
        input_image = data[0][0][::4, ::4, :]
        cv2.imwrite('input.png', input_image[:, :, ::-1])
        print(data[7][0])
        print(data[7][0].flatten())
        inputs = data[0]
        for i, item in enumerate(inputs):
            inputs[i] = item[::4, ::4, :]
        res = sess.run([transform_op], feed_dict={input_images: inputs,
																		input_score_maps: data[2],
																		input_geo_maps: data[3],
																		input_training_masks: data[4],
																		valid_labels: data[5],
																		seq_len: data[6],
																		valid_affines: data[7],
																		text_masks: data[8],
																		is_training: False})
        a = np.ones((9))
        a[:8] = data[7][0]
        
        cv2.imwrite("res_cv2_mask.png", data[8][0]*cv2.warpAffine(input_image[:, :, ::-1], np.linalg.inv(a.reshape(3,3))[:2,:], (160, 160)))
        #cv2.imwrite("res_cv2.png", cv2.warpAffine(input_image[:, :, ::-1], np.linalg.inv(data[7][0])[:2,:], (160, 8)))
        print(res[0].shape)
        cv2.imwrite("res_tf.png", np.squeeze(res[0][0])[:,:,::-1]*data[8][0])


if __name__ == '__main__':
    tf.app.run()
    #main()

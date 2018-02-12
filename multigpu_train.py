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

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))
print "checkpoint path {}".format(FLAGS.checkpoint_path)

def tower_loss(images, score_maps, geo_maps, training_masks, valid_affines, seq_len, mask, targets, reuse_variables=None, is_training=True):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry, logits, text_proposals, before_rotate, image_proposal = model.model(images, valid_affines, seq_len, mask, is_training=is_training)

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
        rotate_image_sum = tf.summary.image('input/rotated_image', image_proposal)
        before_rotate_feature_sum = tf.summary.image('input/rotated_feature_original', before_rotate[:, :, :, 0:1])

    return total_loss, model_loss, [model_loss_sum, total_loss_sum, input_sum, score_map_sum, score_map_pred_sum, geo_map_sum, geo_map_pred_sum, detection_mask_sum, recongnition_mask_sum, rotate_feature_sum, before_rotate_feature_sum, rotate_image_sum] + loss_sum_list, [model_val_sum, total_val_sum]


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

    input_images = tf.placeholder(tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
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
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss, train_sum, test_sum = tower_loss(iis, isms, igms, itms, valid_affines, seq_len, text_masks, valid_labels, reuse_variables, is_training)
                train_sums.extend(train_sum)
                test_sums.extend(test_sum)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # summary_op = tf.summary.merge_all()
    tr_summary_op = tf.summary.merge(train_sums + [lr_sum])
    te_summary_op = tf.summary.merge(test_sums)
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, tf.get_collection(tf.GraphKeys.MODEL_VARIABLES),
                                                             ignore_missing_vars=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        val_data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus),
                                         val=True)

        start = time.time()
        for step in range(1, FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, tr_summary_op], feed_dict={input_images: data[0],
                                                                                input_score_maps: data[2],
                                                                                input_geo_maps: data[3],
                                                                                input_training_masks: data[4],
                                                                                valid_labels: data[5],
                                                                                seq_len: data[6],
                                                                                valid_affines: data[7],
                                                                                text_masks: data[8],
                                                                                is_training: True})

            summary_writer.add_summary(summary_str, global_step=step)

            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))
            if step % FLAGS.save_summary_steps == 0:
                val_data = next(val_data_generator)
                ml_val, tl_val, test_sum = sess.run([model_loss, total_loss, te_summary_op], feed_dict={input_images: val_data[0],
                                                                                    input_score_maps: val_data[2],
                                                                                    input_geo_maps: val_data[3],
                                                                                    input_training_masks: val_data[4],
                                                                                    valid_labels: val_data[5],
                                                                                    seq_len: val_data[6],
                                                                                    valid_affines: val_data[7],
                                                                                    text_masks: val_data[8],
                                                                                    is_training: False})
                print('Step {:06d}, val model loss {:.4f}, val total loss {:.4f}'.format(
                    step, ml_val, tl_val))
                summary_writer.add_summary(test_sum, global_step=step)

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

if __name__ == '__main__':
    tf.app.run()

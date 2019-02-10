import math

import tensorflow as tf
from tqdm import tqdm as progress


class ThreeLayerCNN:
    __slots__ = ('_subimage_provider', '_input_transform', '_target_transform', '_batch_size', '_weights_stdev',
                 '_beta', 'session', '_net_input', '_net_target', 'network', '_loss_function', '_train_step',
                 '_training_summaries', '_evaluation_summaries', '_log_training_summary', '_log_evaluation_summary',
                 '_save_file', '_saver')

    def __init__(self, subimage_provider, input_tranform, target_transform,
                 batch_size=100, input_bands_count=21, target_bands_count=3,
                 first_layer_shape=(5, 5, 64), second_layer_shape=(5, 5, 32), third_layer_shape=(5, 5, 3),
                 weights_stdev=0.001, beta=0.1, learning_rate=0.0001, save_file=None):
        self._subimage_provider = subimage_provider
        self._input_transform = input_tranform
        self._target_transform = target_transform
        self._batch_size = batch_size
        self._weights_stdev = weights_stdev
        self._beta = beta
        self.session = tf.Session()
        self._save_file = save_file
        self._training_summaries = []
        self._evaluation_summaries = []

        # Normalize data
        net_input, net_target = self._initialize_placeholders(input_bands_count, target_bands_count)
        norm_input, norm_target = self._normalize_placeholder_data(net_input, net_target)

        # Build network layers
        fl_kernel_height, fl_kernel_width, fl_feature_count = first_layer_shape
        first_layer, first_layer_weights = self._declare_first_layer(
            norm_input, (fl_kernel_height, fl_kernel_width, input_bands_count, fl_feature_count))

        sl_kernel_height, sl_kernel_width, sl_feature_count = second_layer_shape
        second_layer, second_layer_weights = self._declare_second_layer(
            first_layer, (sl_kernel_height, sl_kernel_width, fl_feature_count, sl_feature_count))

        tl_kernel_height, tl_kernel_width, tl_feature_count = third_layer_shape
        third_layer, third_layer_weights = self._declare_third_layer(
            second_layer, (tl_kernel_height, tl_kernel_width, sl_feature_count, tl_feature_count))

        # Declare loss function
        regularization_weights = [first_layer_weights, second_layer_weights, third_layer_weights]
        loss_function = self._declare_loss_function(third_layer, norm_target, regularization_weights)
        train_step = self._declare_optimizer_train_step(learning_rate, loss_function)

        self._net_input = net_input
        self._net_target = net_target
        self.network = third_layer
        self._loss_function = loss_function
        self._train_step = train_step

        self._log_training_summary = tf.summary.merge(self._training_summaries)
        self._log_evaluation_summary = tf.summary.merge(self._evaluation_summaries)

        graph_initialization = tf.global_variables_initializer()
        self.session.run(graph_initialization)

        self._saver = tf.train.Saver()


    def train_model(self, epochs=10000, summary_every=25, log_writer=None):
        subimage_provider = self._subimage_provider
        evaluation_input, evaluation_target = subimage_provider.evaluation_subimages()
        evaluation_feed_dict = {self._net_input: evaluation_input, self._net_target: evaluation_target}

        updates_per_epoch = int(math.ceil(subimage_provider.training_subimage_count() / self._batch_size))

        sess = self.session
        training_loss_values = []
        evaluation_loss_values = []
        min_train_loss = 10e3
        min_eval_loss = 10e3
        for e_num, epoch in enumerate(progress(range(epochs), desc='Training model...', unit='epochs')):
            for u_num, update in enumerate(range(updates_per_epoch)):
                input_batch, target_batch = subimage_provider.random_training_subimage_batch(
                    batch_size=self._batch_size)
                training_feed_dict = {self._net_input: input_batch, self._net_target: target_batch}
                sess.run(self._train_step, feed_dict=training_feed_dict)

                if (e_num + 1) % summary_every == 0 and u_num == updates_per_epoch - 1:
                    training_loss_value = sess.run(self._loss_function, feed_dict=training_feed_dict)
                    training_loss_values.append(training_loss_value)
                    evaluation_loss_value = sess.run(self._loss_function, feed_dict=evaluation_feed_dict)
                    evaluation_loss_values.append(evaluation_loss_value)
                    print('\nEpoch #: {} -> Training Loss: {}, Evaluation Loss: {}'
                          .format(e_num + 1, str(training_loss_value), str(evaluation_loss_value)))

                    if log_writer is not None:
                        self._write_training_logs(log_writer, e_num, training_feed_dict, evaluation_feed_dict)

                    if self._save_file is not None and \
                            evaluation_loss_value < min_eval_loss and training_loss_value < min_train_loss:
                        min_train_loss = training_loss_value
                        min_eval_loss = evaluation_loss_value
                        self._saver.save(sess, self._save_file)
        print('\nMin training Loss: {}, Min evaluation Loss: {}'.format(str(min_train_loss), str(min_eval_loss)))
        return training_loss_values, evaluation_loss_values

    def evaluate(self, input_data):
        sess = self.session
        if self._save_file:
            self._saver.restore(sess, self._save_file)

        feed_dict = {self._net_input: input_data}
        return sess.run(self._target_transform.denormalize(self.network), feed_dict=feed_dict)

    #
    # Private methods
    #

    def _initialize_placeholders(self, input_bands_count, target_bands_count):
        image_width, image_height = self._subimage_provider.subimage_shape()

        input_shape = (None, image_width, image_height, input_bands_count)
        target_shape = (None, image_width, image_height, target_bands_count)

        net_input = tf.placeholder(tf.uint16, shape=input_shape, name='net_input')
        self._training_summaries.append(tf.summary.histogram('net_input', net_input))

        net_target = tf.placeholder(tf.uint16, shape=target_shape, name='net_target')
        self._training_summaries.append(tf.summary.histogram('net_target', net_target))

        return net_input, net_target

    def _normalize_placeholder_data(self, net_input, net_target):
        with tf.name_scope('input_normalization'):
            norm_input = self._input_transform.normalize(net_input)
            self._training_summaries.append(tf.summary.histogram('norm_input', norm_input))

        with tf.name_scope('target_normalization'):
            norm_target = self._target_transform.normalize(net_target)
            self._training_summaries.append(tf.summary.histogram('norm_target', norm_target))

        return norm_input, norm_target

    def _declare_first_layer(self, norm_input, layer_shape):
        return self._declare_layer(norm_input, layer_shape, with_relu=True, layer_name='layer_1')

    def _declare_second_layer(self, first_layer, layer_shape):
        return self._declare_layer(first_layer, layer_shape, with_relu=True, layer_name='layer_2')

    def _declare_third_layer(self, second_layer, layer_shape):
        return self._declare_layer(second_layer, layer_shape, with_relu=False, layer_name='layer_3')

    def _declare_layer(self, layer_input, layer_shape, with_relu, layer_name):
        with tf.name_scope(layer_name):
            conv_weights = tf.Variable(tf.truncated_normal(layer_shape, stddev=self._weights_stdev, dtype=tf.float32),
                                       name=layer_name + '_weights')
            self._training_summaries.append(tf.summary.histogram('weights', conv_weights))

            kernel_height, kernel_width, _, _ = layer_shape
            height_pad = kernel_height // 2
            width_pad = kernel_width // 2
            padded_input = tf.pad(layer_input,
                                  [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]], mode='SYMMETRIC')
            conv = tf.nn.conv2d(padded_input, conv_weights, strides=[1, 1, 1, 1], padding='VALID')

            if not with_relu:
                self._training_summaries.append(tf.summary.histogram('output', conv))
                return conv, conv_weights

            _, _, _, feature_count = layer_shape
            relu_bias = tf.Variable(tf.zeros((feature_count,), dtype=tf.float32), name=layer_name + '_bias')
            self._training_summaries.append(tf.summary.histogram('bias', relu_bias))
            relu = tf.nn.relu(tf.nn.bias_add(conv, relu_bias))
            self._training_summaries.append(tf.summary.histogram('output', relu))
            return relu, conv_weights

    def _declare_loss_function(self, network, net_target, regularization_weights):
        with tf.name_scope('loss'):
            error = tf.squared_difference(network, net_target)
            loss = tf.reduce_mean(tf.reduce_sum(error, [1, 2, 3]))

            def recursive_l2_loss_sum(elem, array):
                regularized_elem = tf.nn.l2_loss(elem)
                if len(array) == 0:
                    return regularized_elem
                if len(array) == 1:
                    return regularized_elem + recursive_l2_loss_sum(array[0], [])
                return regularized_elem + recursive_l2_loss_sum(array[0], array[1:])

            regularizers = recursive_l2_loss_sum(regularization_weights[0], regularization_weights[1:])
            regularized_loss = tf.reduce_mean(loss + self._beta * regularizers)

            self._training_summaries.append(tf.summary.scalar('training_loss', regularized_loss))
            self._evaluation_summaries.append(tf.summary.scalar('evaluation_loss', regularized_loss))
            return regularized_loss

    def _declare_optimizer_train_step(self, learning_rate, loss_function):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        return optimizer.minimize(loss_function)

    def _write_training_logs(self, log_writer, epoch_num, training_feed_dict, evaluation_feed_dict):
        sess = self.session

        training_summary = sess.run(self._log_training_summary, feed_dict=training_feed_dict)
        log_writer.add_summary(training_summary, epoch_num)

        evaluation_summary = sess.run(self._log_evaluation_summary, feed_dict=evaluation_feed_dict)
        log_writer.add_summary(evaluation_summary, epoch_num)

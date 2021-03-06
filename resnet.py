from collections import namedtuple

import tensorflow as tf
import numpy as np


import utils_wrn as utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, '
                    'weight_decay, initial_lr, decay_step, decay_rate, '
                    'momentum, drop_prob')
                    
   
 

class WResNet(object):
    def __init__(self, hp, images, labels, test_labels,global_step):#, is_train
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
        self._test_labels = test_labels
        self._global_step = global_step
        self.is_train = tf.placeholder(tf.bool)        

        
    def network(self):
        print('network')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, 16, 1, name='init_conv')

        # Residual Blocks
        filters = [16, 16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k]
        strides = [1, 2, 2]

        for i in range(1, 4): #(1, 4)
            # First residual unit
            with tf.variable_scope('unit_%d_0' % i) as scope:
                print('\tBuilding residual unit: %s' % scope.name)
                x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                x = utils._relu(x, name='relu_1')

                # Shortcut
                if filters[i-1] == filters[i]:
                    if strides[i-1] == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides[i-1], strides[i-1], 1],
                                                  [1, strides[i-1], strides[i-1], 1], 'VALID')
                else:
                    shortcut = utils._conv(x, 1, filters[i], strides[i-1], name='shortcut')

                # Residual
                x = utils._conv(x, 3, filters[i], strides[i-1], name='conv_1')
                x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                x = utils._relu(x, name='relu_2')
                x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                # Merge
                x = x + shortcut
            # Other residual units
            for j in range(1, self._hp.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    # Shortcut
                    shortcut = x

                    # Residual
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                    x = utils._relu(x, name='relu_1')
                    x = utils._conv(x, 3, filters[i], 1, name='conv_1')
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                    x = utils._relu(x, name='relu_2')
                    x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                    # Merge
                    x = x + shortcut

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            x = utils._fc(x, (self._hp.num_classes-1)*2)

        self._logits = x

        # Probs & preds & acc
        #self.probs = tf.nn.softmax(x, name='probs')
        #self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        #self.preds = tf.cast(tf.argmax(self._logits, 1, name='preds'),dtype=tf.int32)


    def build_model(self):
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, 16, 1, name='init_conv')

        # Residual Blocks
        filters = [16, 16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k]
        strides = [1, 2, 2]

        for i in range(1, 3):#(1,4)
            # First residual unit
            with tf.variable_scope('unit_%d_0' % i) as scope:
                print('\tBuilding residual unit: %s' % scope.name)
                x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                x = utils._relu(x, name='relu_1')

                # Shortcut
                if filters[i-1] == filters[i]:
                    if strides[i-1] == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides[i-1], strides[i-1], 1],
                                                  [1, strides[i-1], strides[i-1], 1], 'VALID')
                else:
                    shortcut = utils._conv(x, 1, filters[i], strides[i-1],  name='shortcut')# pad='VALID',

                # Residual
                #x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                #x = utils._relu(x, name='relu_1')
                x = utils._conv(x, 3, filters[i], strides[i-1], name='conv_1')
                x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                x = utils._relu(x, name='relu_2')
                x = tf.layers.dropout(x, rate=self._hp.drop_prob, training=True, name='dropout')
                x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                # Merge
                x = x + shortcut
            # Other residual units
            for j in range(1, self._hp.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    # Shortcut
                    shortcut = x

                    # Residual
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                    x = utils._relu(x, name='relu_1')
                    x = utils._conv(x, 3, filters[i], 1, name='conv_1')
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                    x = utils._relu(x, name='relu_2')
                    x = tf.layers.dropout(x, rate=self._hp.drop_prob, training=True, name='dropout')
                    x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                    # Merge
                    x = x + shortcut

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])
            

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            x = utils._fc(x, (self._hp.num_classes-1)*2)

        self._logits = x

        # Probs & preds & acc
        self.preds = tf.reshape(self._logits, [-1, self._hp.num_classes-1, 2])
        self.preds = tf.argmax(self.preds, axis=2)
        self.preds = tf.cast(tf.reduce_sum(self.preds, axis=1, name='preds'), dtype=tf.int32)
        #self.preds = tf.cast(tf.argmax(self._logits, 1, name='preds'),dtype=tf.int32)

        ones = tf.cast(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._test_labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)

        # Loss & acc
        outs = []
        for i in range(self._hp.num_classes-1):
            age_label = tf.slice(self._labels, [0,i], [self._hp.batch_size,1])
            age_label = tf.reshape(age_label, [self._hp.batch_size])
            age_fc1 = tf.slice(x, [0, 2*i], [self._hp.batch_size, 2])
            age_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=age_fc1, labels=age_label)
            outs.append(age_softmax)
            #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
        loss = tf.add_n(outs)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar('cross_entropy', self.loss)


    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            # for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
                # tf.summary.histogram(var.op.name, var)
            l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
        self._total_loss = self.loss + l2_loss

        # Learning rate
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        self._hp.decay_step, self._hp.decay_rate, staircase=True)
        tf.summary.scalar('learing_rate', self.lr)

        # Gradient descent step
        #opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        opt = tf.train.AdamOptimizer(self.lr)
        #opt = tf.contrib.opt.MovingAverageOptimizer(opt, average_decay = 0.999)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(self._total_loss, global_step=self._global_step)
        
#        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
#        # print('\n'.join([t.name for t in tf.trainable_variables()]))
#        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

#        # Batch normalization moving average update
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#        if update_ops:
#            with tf.control_dependencies(update_ops+[apply_grad_op]):
#                self.train_op = tf.no_op()  #tf.no_op mean do nothing
#        else:
#            self.train_op = apply_grad_op

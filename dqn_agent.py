import numpy as np
from memory import Memory
from action_discretization import Action_discretization
import tensorflow as tf
import tensorflow.contrib.layers as layers
from config import abspath



# the class contains the network topology of the DQN and experience replay method
class Dqn_agent:
    def __init__(self, asset_num, division, feature_num, gamma,
                 network_topology,
                 learning_rate,
                 epsilon, epsilon_Min,
                 epsilon_decay_period,
                 update_tar_period,
                 history_length,
                 memory_size,
                 batch_size,
                 save_period,
                 name,
                 save):

        self.epsilon = epsilon
        self.epsilon_min = epsilon_Min
        self.epsilon_decay_period = epsilon_decay_period
        self.asset_num = asset_num
        self.division = division
        self.gamma = gamma
        self.name = name
        self.update_tar_period = update_tar_period
        self.history_length = history_length
        self.feature_num = feature_num
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = learning_rate
        self.cnn_trainable=True
        self.action_num, self.actions = action_discretization(self.asset_num, self.division)
        config = tf.ConfigProto()


        self.sess = tf.Session(config=config)

        network_topology['output_num'] = self.action_num

        self.network_config = network_topology
        self.initialize_graph()
        t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='estm_net')

        # assign parameters of estimate Q-net to target Q-net
        self.update_target = [tf.assign(t, l) for t, l in zip(t_params, e_params)]
        self.sess.run(tf.global_variables_initializer())
        self.memory = Memory(self.action_num, self.actions, memory_size=memory_size, batch_size=batch_size)

        if save:
            self.save = save
            self.save_period = save_period
            self.name = name
            self.saver = tf.train.Saver()
        else:
            self.save = False


    # initialize variables that will be used in the training process
    def initialize_graph(self):
        # current price tensor
        self.price_his = tf.placeholder(dtype=tf.float32,
                                        shape=[None, self.asset_num - 1, self.history_length, self.feature_num],
                                        name="ob")

        # price tensor of next step
        self.price_his_ = tf.placeholder(dtype=tf.float32,
                                         shape=[None, self.asset_num - 1, self.history_length, self.feature_num],
                                         name="ob_")

        # weight vector of current step
        self.addi_inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs')

        # weight vector of next step
        self.addi_inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, self.asset_num], name='addi_inputs_')

        # the actions chose by the DQN agent
        self.a = tf.placeholder(dtype=tf.int32, shape=[None, ], name='a')
        self.input_num = tf.placeholder(dtype=tf.int32, shape=[])

        # weight of each memory from the memory pool
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        # Q-values of extimate net
        with tf.variable_scope('estm_net'):
            self.fc_input, self.q_pred = self.build_graph(self.price_his, self.addi_inputs,self.cnn_trainable)

        # Q-values of target net
        with tf.variable_scope('target_net'):
            _,self.tar_pred = self.build_graph(self.price_his_, self.addi_inputs_, self.cnn_trainable)

        # a holder to contain the target Q-value
        with tf.variable_scope('q_tar'):
            self.q_target = tf.placeholder(dtype=tf.float32, shape=[None], name='q_target')

        # select the largest estimate Q-values
        with tf.variable_scope('q_estm_wa'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_estm_wa = tf.gather_nd(params=self.q_pred, indices=a_indices)

        # loss function
        with tf.name_scope('loss'):
            error = tf.abs(self.q_target-self.q_estm_wa)
            self.abs_errors = error
            square = tf.square(error)
            self.loss = tf.reduce_mean(self.ISWeights*square)

        # update the parameters of estimate Q-net
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


    # network topology
    def build_graph(self, price_his, addi_input, trainable):
        kernels = self.network_config['kernels']
        strides = self.network_config['strides']
        filters = self.network_config['filters']
        fc1_size = self.network_config['fc1_size']

        # choose activate function
        def set_activation(activation):
            if activation == 'relu':
                activation = tf.nn.relu
            elif activation == 'selu':
                activation = tf.nn.selu
            else:
                activation = tf.nn.leaky_relu
            return activation

        cnn_activation = set_activation(self.network_config['cnn_activation'])
        w_initializer = tf.random_uniform_initializer(-0.05, 0.05)
        b_initializer = tf.constant_initializer(self.network_config['b_initializer'])
        regularizer = layers.l2_regularizer(self.network_config['regularizer'])

        conv = price_his

        # first cnn layer
        conv = tf.layers.conv2d(conv, filters=filters[0], kernel_size=kernels[0], strides=strides[0],
                                     trainable=trainable, activation=cnn_activation,
                                     kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                     kernel_initializer= w_initializer , bias_initializer=b_initializer,
                                     padding='same', name=self.name+'conv'+str(0))

       # second cnn layer
        conv = tf.layers.conv2d(conv, filters=filters[1], kernel_size=kernels[1], strides=strides[1],
                                trainable=trainable, activation=cnn_activation,
                                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                kernel_initializer=w_initializer, bias_initializer=b_initializer,
                                padding='same', name=self.name + 'conv' + str(1))

        # weight vector with the weight of cash removed
        addi_input1 = addi_input[:, 1:]

        # insert weight vector into the feature maps
        conv = tf.concat([conv, addi_input1[:, :, np.newaxis, np.newaxis]], axis=3)

        # third cnn layer
        conv =  tf.layers.conv2d(conv, filters=filters[2], kernel_size=kernels[2], strides=strides[2],
                                trainable=trainable, activation=cnn_activation,
                                kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                kernel_initializer=w_initializer, bias_initializer=b_initializer,
                                padding='same', name=self.name + 'conv' + str(2))

        cash_bias = tf.ones((self.input_num, 1))

        conv = tf.layers.flatten(conv)

        fc_input = tf.concat([cash_bias, conv], 1)

        fc1 = layers.fully_connected(fc_input, num_outputs=fc1_size, activation_fn= None,
                                     weights_initializer=w_initializer,
                                     trainable=True, scope=self.name+'fc1')

        output_state=layers.fully_connected(fc1, num_outputs=1, activation_fn=None,
                                        weights_initializer=w_initializer,
                                        trainable=True, scope=self.name+'output_state')

        output_action = layers.fully_connected(fc1, num_outputs=self.action_num, activation_fn=None,
                                        weights_initializer=w_initializer,
                                       trainable=True, scope=self.name+'output_action')

        output = output_state + (output_action - tf.reduce_mean(output_action, axis=1, keep_dims=True))

        return fc_input, output




    def replay(self):

        obs, action_batch, reward_batch, obs_, tree_idx, ISWeights = self.memory.sample()

        q_values_next = self.sess.run(self.q_pred, feed_dict={self.price_his: obs_['history'],
                                                              self.addi_inputs:obs_['weights'],
                                                              self.input_num:obs_['history'].shape[0]})

        best_actions = np.argmax(q_values_next, axis=1)

        q_values_next_target = self.sess.run(self.tar_pred, feed_dict={self.price_his_: obs_['history'],
                                                                       self.addi_inputs_:obs_['weights'],
                                                                       self.input_num: obs_['history'].shape[0]})

        targets_batch = reward_batch + self.gamma * q_values_next_target[np.arange(len(action_batch)), best_actions]

        fd = {self.q_target: targets_batch,
              self.price_his: obs['history'],
              self.addi_inputs: obs['weights'],
              self.a : action_batch,
              self.input_num: obs['history'].shape[0],
              self.ISWeights:ISWeights}

        _, abs_errors, global_step = self.sess.run([self.train_op, self.abs_errors, self.global_step], feed_dict=fd)

        self.memory.batch_update(tree_idx, abs_errors)

        if global_step % self.update_tar_period == 0:
            self.sess.run(self.update_target)

        if self.save and global_step % self.save_period == 0:
            self.saver.save(self.sess, abspath+'logs/checkpoint/' + self.name, global_step=global_step)

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1 - self.epsilon_min) / self.epsilon_decay_period



    def choose_action(self, observation, test):

        def action_max():
            fc_input,action_values = self.sess.run([self.fc_input,self.q_pred],
                                                  feed_dict={self.price_his: observation['history'][np.newaxis, :, :, :],
                                                  self.addi_inputs: observation['weights'][np.newaxis, :],
                                                  self.input_num : 1})
            return np.argmax(action_values), fc_input

        if not test:
            if np.random.rand() > self.epsilon:
                action_idx, fc_input= action_max()
            else:
                action_idx = np.random.randint(0, self.action_num)
                action_idx_, fc_input=action_max()
        else:
            action_idx,fc_input = action_max()

        action_weights = self.actions[action_idx]

        return action_idx, action_weights, fc_input


    def store(self, ob, a, r, ob_):
        self.memory.store(ob, a, r, ob_)

    def get_training_step(self):
        a = self.sess.run(self.global_step)
        return a

    def restore(self, name):
        self.saver.restore(self.sess, abspath+'logs/checkpoint/'+name)

    def start_replay(self):
        return self.memory.start_replay()

    def memory_cnt(self):
        return self.memory.tree.data_pointer

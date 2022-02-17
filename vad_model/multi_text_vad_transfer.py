# Variational Auto-Denoiser with Multinomial output
# For textual processing:
#     embedding layer with mlp hidden layers
import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


class MultiTextVADTransfer(object):
    """
    Inference model:
        Q_phi(z|x, y): to be specified
        Q_psi(y|x): structure given

    Generation model:
        P_theta(x|z, y): to be specified
        P_gamma(wy|z, y): try log linear
    """

    def __init__(self, opt, device, output_bias=None, alpha=1, beta=1, lam=0.01):

        self.device = device
        self.pretrained_size = opt['pretrained_size']
        self.num_classes = opt['num_classes']
        if output_bias is None:
            self.bias_init = np.zeros(self.num_classes)
        else:
            self.bias_init = output_bias
        self.sequence_length = 100

        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.lr = opt['lr']
        self.random_seed = opt['seed']
        self.p_theta_dims = opt['p_theta']
        self.p_gamma_dims = opt['p_gamma']
        self.q_phi_dims = opt['q_phi']
        self.q_phi_dims[0] = self.q_phi_dims[0] * self.pretrained_size
        self.q_psi_dims = opt['q_psi']
        self.q_psi_dims[0] = self.q_psi_dims[0] * self.pretrained_size
        self.x_dims = self.q_phi_dims[:-1] + self.p_theta_dims

        self.construct_placeholders()

    def construct_placeholders(self):
        # data input: word index
        self.input_ph = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[None, self.sequence_length])

        # data input: sentence representations
        self.input_emb = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, self.pretrained_size])

        # multihot data input
        vocab_size = self.p_theta_dims[-1]
        self.input_ph_multihot = tf.reduce_sum(
            tf.one_hot(indices=self.input_ph, depth=vocab_size), axis=1)

        # per batch prior
        self.prior = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[self.num_classes])

        # weak labels: one-hot
        self.weak_labels = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, self.num_classes])

        # training related parameters
        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        # calculate negative log likelihood of both p_theta and p_gamma nets
        saver, logits_theta, logits_gamma, sampled_z, q_y, KL_phi, KL_psi = \
            self.forward_pass()

        log_softmax_var_theta = tf.nn.log_softmax(logits_theta)
        neg_ll_theta = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var_theta * self.input_ph_multihot,
            axis=-1))

        gamma_y = tf.nn.softmax(logits_gamma)
        log_softmax_var_gamma = tf.nn.log_softmax(logits_gamma)
        neg_ll_gamma = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var_gamma * self.weak_labels, axis=-1))

        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights_q_phi + self.weights_q_psi
                                       + self.weights_p_theta + self.weights_p_gamma)

        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll_gamma + self.alpha * KL_psi + \
                   self.beta * neg_ll_theta + KL_phi + 2 * reg_var

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.compat.v1.summary.scalar('negative_ll_theta', neg_ll_theta)
        tf.compat.v1.summary.scalar('KL_phi', KL_phi)
        tf.compat.v1.summary.scalar('negative_ll_gamma', neg_ll_gamma)
        tf.compat.v1.summary.scalar('KL_psi', KL_psi)
        tf.compat.v1.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.compat.v1.summary.merge_all()

        return saver, sampled_z, q_y, gamma_y, neg_ELBO, train_op, merged

    def q_psi_graph(self, h):
        logits_psi, q_y, KL_psi = None, None, None

        # Hidden layers
        with tf.device(self.device), tf.compat.v1.variable_scope("Qpsi"):
            for i, (w, b) in enumerate(zip(self.weights_q_psi, self.biases_q_psi)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_q_psi) - 1:
                    h = tf.nn.tanh(h)
                else:
                    logits_psi = h
                    q_y = tf.nn.softmax(logits_psi)
                    log_q_y = tf.nn.log_softmax(logits_psi)
                    KL_psi = tf.reduce_mean(tf.reduce_sum(
                        self.prior * (tf.math.log(self.prior) - log_q_y), axis=1))

        return logits_psi, q_y, KL_psi

    def q_phi_graph(self, h, logits_psi):
        mu_q, std_q, KL_phi = None, None, None

        # Hidden layers
        with tf.device(self.device), tf.compat.v1.variable_scope("Qphi"):
            h = tf.concat([h, logits_psi], 1)
            for i, (w, b) in enumerate(zip(self.weights_q_phi, self.biases_q_phi)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_q_phi) - 1:
                    h = tf.nn.tanh(h)
                else:
                    mu_q = h[:, :self.q_phi_dims[-1]]
                    logvar_q = h[:, self.q_phi_dims[-1]:]

                    std_q = tf.exp(0.5 * logvar_q)
                    KL_phi = tf.reduce_mean(tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))
        return mu_q, std_q, KL_phi

    def p_theta_graph(self, z, logits_psi):
        with tf.device(self.device), tf.compat.v1.variable_scope("PTheta"):
            h = tf.concat([z, logits_psi], 1)

            for i, (w, b) in enumerate(zip(self.weights_p_theta, self.biases_p_theta)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_p_theta) - 1:
                    h = tf.nn.tanh(h)
        return h

    def p_gamma_graph(self, z, logits_psi):
        with tf.device(self.device), tf.compat.v1.variable_scope("PGamm"):
            h = tf.concat([z, logits_psi], 1)

            for i, (w, b) in enumerate(zip(self.weights_p_gamma, self.biases_p_gamma)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_p_gamma) - 1:
                    h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # embedding layer
        embedding = tf.nn.dropout(self.input_emb, rate=1 - self.keep_prob_ph)

        # q_psi network
        logits_psi, q_y, KL_psi = self.q_psi_graph(embedding)

        # q_phi network
        mu_q, std_q, KL_phi = self.q_phi_graph(embedding, logits_psi)
        epsilon = tf.random.normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * epsilon * std_q

        # p_theta network
        logits_theta = self.p_theta_graph(sampled_z, logits_psi)

        # p_gamma network
        logits_gamma = self.p_gamma_graph(sampled_z, logits_psi)

        return tf.compat.v1.train.Saver(), logits_theta, logits_gamma, \
               sampled_z, q_y, KL_phi, KL_psi

    def _construct_weights(self):
        """
        Latent feature models
        """
        self.weights_q_phi, self.biases_q_phi = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_phi_dims[:-1], self.q_phi_dims[1:])):
            if i == len(self.q_phi_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)
            weight_initializer = tf.contrib.layers.xavier_initializer(
                seed=self.random_seed)
            bias_initializer = tf.truncated_normal_initializer(
                stddev=0.001, seed=self.random_seed)
            weight_layer = tf.compat.v1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                     initializer=weight_initializer)
            bias_layer = tf.compat.v1.get_variable(name=bias_key, shape=[d_out],
                                                   initializer=bias_initializer)

            if i == 0:
                # the input to z also takes y as input
                class_weight = tf.compat.v1.get_variable(
                    name=weight_key + '_class', shape=[self.num_classes, d_out],
                    initializer=tf.contrib.layers.xavier_initializer(
                        seed=self.random_seed))
                self.weights_q_phi.append(tf.concat([weight_layer, class_weight], axis=0))
            else:
                self.weights_q_phi.append(weight_layer)

            self.biases_q_phi.append(bias_layer)

            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_q_phi[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_q_phi[-1])

        self.weights_p_theta, self.biases_p_theta = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_theta_dims[:-1], self.p_theta_dims[1:])):

            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            weight_initializer = tf.contrib.layers.xavier_initializer(
                seed=self.random_seed)
            bias_initializer = tf.truncated_normal_initializer(
                stddev=0.001, seed=self.random_seed)

            weight_layer = tf.compat.v1.get_variable(name=weight_key, shape=[d_in, d_out],
                                                     initializer=weight_initializer)
            bias_layer = tf.compat.v1.get_variable(name=bias_key, shape=[d_out],
                                                   initializer=bias_initializer)

            if i == 0:
                # the input to z also takes y as input
                class_weight = tf.compat.v1.get_variable(
                    name=weight_key + '_class', shape=[self.num_classes, d_out],
                    initializer=tf.contrib.layers.xavier_initializer(
                        seed=self.random_seed))
                self.weights_p_theta.append(tf.concat([weight_layer, class_weight], axis=0))
            else:
                self.weights_p_theta.append(weight_layer)

            self.biases_p_theta.append(bias_layer)

            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_p_theta[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_p_theta[-1])

        '''
        Latent class models
        '''
        self.weights_q_psi, self.biases_q_psi = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_psi_dims[:-1], self.q_psi_dims[1:])):
            weight_key = "weight_q_psi_{}to{}".format(i, i + 1)
            bias_key = "bias_q_psi_{}".format(i + 1)

            self.weights_q_psi.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            if i == len(self.q_psi_dims) - 2:
                self.biases_q_psi.append(tf.compat.v1.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.constant_initializer(self.bias_init)))
            else:
                self.biases_q_psi.append(tf.compat.v1.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_q_psi[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_q_psi[-1])

        self.weights_p_gamma, self.biases_p_gamma = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_gamma_dims[:-1], self.p_gamma_dims[1:])):
            if i == 0:
                # the input to z also takes y as input
                d_in += self.num_classes

            weight_key = "weight_p_gamma_{}to{}".format(i, i + 1)
            bias_key = "bias_p_gamma_{}".format(i + 1)
            self.weights_p_gamma.append(tf.compat.v1.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            self.biases_p_gamma.append(tf.compat.v1.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001, seed=self.random_seed)))
            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_p_gamma[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_p_gamma[-1])
            # check the influence of latent class
            tf.compat.v1.summary.histogram(
                weight_key + 'latent-contr', self.weights_p_gamma[-1][-1, :])

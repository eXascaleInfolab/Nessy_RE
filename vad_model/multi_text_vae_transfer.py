# Variational Autoencoder with Multinomial output
# For textual processing:
#     embedding layer with mlp hidden layers
# Adapted from Dawen Liang's code

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


class MultiTextVAETransfer():

    def __init__(self, p_dims, q_dims, emb_size, config, lam=0.01, random_seed=31):

        self.device = config["device"]
        self.lr = config["lr"]
        self.emb_size = emb_size
        self.vocab_size = config["vocab_size"]
        self.sequence_length = config["seq_len"]
        self.lam = lam
        self.random_seed = random_seed
        assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
        self.p_dims = p_dims
        self.q_dims = q_dims
        self.q_dims[0] = self.q_dims[0] * self.emb_size
        self.construct_placeholders()

    def construct_placeholders(self):
        self.input_ph = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[None, self.sequence_length])
        self.input_emb = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[None, self.emb_size])
        self.dep_path_ph = tf.compat.v1.placeholder(
            dtype=tf.int32, shape=[None, self.sequence_length])

        self.input_ph_multihot = tf.reduce_sum(
            tf.one_hot(indices=self.dep_path_ph, depth=self.vocab_size), reduction_indices=1)

        self.keep_prob_ph = tf.compat.v1.placeholder_with_default(1.0, shape=None)
        self.is_training_ph = tf.compat.v1.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.compat.v1.placeholder_with_default(1., shape=None)

    def build_graph(self):
        self._construct_weights()

        saver, logits, KL, sampled_z = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph_multihot,
            axis=-1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)

        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var

        train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.compat.v1.summary.scalar('negative_multi_ll', neg_ll)
        tf.compat.v1.summary.scalar('KL', KL)
        tf.compat.v1.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.compat.v1.summary.merge_all()

        return saver, logits, sampled_z, neg_ELBO, train_op, merged

    def q_graph(self):
        mu_q, std_q, KL = None, None, None

        h = tf.nn.dropout(self.input_emb, rate=1-self.keep_prob_ph)

        # Hidden layers
        with tf.device(self.device), tf.compat.v1.variable_scope("Qhidden"):
            for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_q) - 1:
                    h = tf.nn.tanh(h)
                else:
                    mu_q = h[:, :self.q_dims[-1]]
                    logvar_q = h[:, self.q_dims[-1]:]

                    std_q = tf.exp(0.5 * logvar_q)
                    KL = tf.reduce_mean(tf.reduce_sum(
                            0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        with tf.device(self.device), tf.compat.v1.variable_scope("Phidden"):
            h = z

            for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
                h = tf.matmul(h, w) + b

                if i != len(self.weights_p) - 1:
                    h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random.normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph *\
            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf.compat.v1.train.Saver(), logits, KL, sampled_z

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)

            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))

            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_q[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_q[-1])

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))

            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))

            # add summary stats
            tf.compat.v1.summary.histogram(weight_key, self.weights_p[-1])
            tf.compat.v1.summary.histogram(bias_key, self.biases_p[-1])


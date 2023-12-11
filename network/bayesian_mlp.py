import tensorflow.compat.v1 as tf
import math


class Bayesian_FC:
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False,
                 clip_min = -10.0, clip_max = 1.0, sigma_bias=-2.0):
        
        with tf.variable_scope(name, reuse=reuse):
            mu_w = tf.get_variable("mu_w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            logsig_w = tf.get_variable("logsig_w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            
            mu_b = tf.get_variable("mu_b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)
            logsig_b = tf.get_variable("logsig_b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)
            

            sig_w = tf.exp(tf.clip_by_value(logsig_w, clip_min, clip_max) + sigma_bias)
            sig_b = tf.exp(tf.clip_by_value(logsig_b, clip_min, clip_max) + sigma_bias)

            dist_w = tf.distributions.Normal(mu_w, sig_w)
            dist_b = tf.distributions.Normal(mu_b, sig_b)
            dist_prior = tf.distributions.Normal(tf.zeros_like(mu_w), tf.exp(tf.zeros_like(sig_w) + sigma_bias))

            out = tf.matmul(input_tensor, dist_w.sample()) / (input_dim ** 0.5) + dist_b.sample()
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            
            self.layer_output = out
            self.layer_mean = tf.matmul(input_tensor, mu_w) / (input_dim ** 0.5) + mu_b 
            self.layer_var = tf.matmul(input_tensor ** 2, sig_w ** 2)  / (input_dim ** 0.5) + sig_b ** 2

            self.regularization_loss = tf.reduce_mean(dist_w.kl_divergence(dist_prior)) + tf.reduce_mean(dist_b.kl_divergence(dist_prior))
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)


class FC:
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False,):
        
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            b = tf.get_variable("b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)


            out = tf.matmul(input_tensor, w) / (input_dim ** 0.5) + b
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            
            self.layer_output = out
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            self.regularization_loss = tf.reduce_mean(w ** 2 + b ** 2)

class Variational_FC:
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False, 
                 clip_min = -10.0, clip_max = 1.0, sigma_bias=-2.0):
        
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("w", shape=[input_dim, output_dim * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            b = tf.get_variable("b", shape=[output_dim * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)


            out = (tf.matmul(input_tensor, w) + b) / (input_dim ** 0.5)

            self.mu, self.logsig = tf.split(out, 2, axis=1)
            self.sig = tf.exp(tf.clip_by_value(self.logsig, clip_min, clip_max) + sigma_bias)

            self.dist = tf.distributions.Normal(self.mu, self.sig)
            self.dist_prior = tf.distributions.Normal(tf.zeros_like(self.mu), tf.exp(tf.zeros_like(self.sig) + sigma_bias))

            out = self.dist.sample()
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            self.layer_output = out
            self.var = self.sig ** 2
            self.regularization_loss = tf.reduce_mean(self.dist.kl_divergence(self.dist_prior))
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
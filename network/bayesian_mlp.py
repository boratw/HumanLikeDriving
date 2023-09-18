import tensorflow.compat.v1 as tf
import math


class Bayesian_FC:
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False,
                 clip_min = -10.0, clip_max = 1.0):
        
        with tf.variable_scope(name, reuse=reuse):
            mu_w = tf.get_variable("mu_w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            logsig_w = tf.get_variable("logsig_w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            
            mu_b = tf.get_variable("mu_b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)
            logsig_b = tf.get_variable("logsig_b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)
            
            noise_w = tf.random.normal([input_dim, output_dim])
            noise_b = tf.random.normal([output_dim])

            sig_w = tf.exp(tf.clip_by_value(logsig_w - 2., clip_min, clip_max))
            sig_b = tf.exp(tf.clip_by_value(logsig_b - 2., clip_min, clip_max))

            w = mu_w + sig_w * noise_w
            b = mu_b + sig_b * noise_b

            out = (tf.matmul(input_tensor, w) + b) / (input_dim ** 0.5)
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            
            self.layer_output = out
            self.layer_mean = (tf.matmul(input_tensor, mu_w) + mu_b) / (input_dim ** 0.5)
            self.layer_var = (tf.matmul(input_tensor ** 2, sig_w ** 2) + sig_b ** 2 ) / (input_dim ** 0.5)
            self.regularization_loss = tf.reduce_mean(mu_w ** 2 + logsig_w ** 2) + tf.reduce_mean(mu_b ** 2 + logsig_b ** 2)
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)


class FC:
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False,):
        
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("w", shape=[input_dim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            b = tf.get_variable("b", shape=[output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)


            out = (tf.matmul(input_tensor, w) + b) / (input_dim ** 0.5)
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
    def __init__(self, input_tensor, input_dim, output_dim, input_dropout= None, output_nonln=None, name=None, reuse=False,):
        
        with tf.variable_scope(name, reuse=reuse):
            w = tf.get_variable("w", shape=[input_dim, output_dim * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(input_dim + output_dim), 1.0 / math.sqrt(input_dim + output_dim), dtype=tf.float32), trainable=True)
            b = tf.get_variable("b", shape=[output_dim * 2], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(output_dim), 1.0 / math.sqrt(output_dim), dtype=tf.float32), trainable=True)


            out = (tf.matmul(input_tensor, w) + b) / (input_dim ** 0.5)
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            self.mu, logsig = tf.split(out, 2, axis=1)
            self.logsig = tf.clip_by_value(logsig, -10, 2) - 2.
            self.sig = tf.exp(self.logsig)

            noise = tf.random.normal(tf.shape(self.mu))
            
            self.layer_output = self.mu + self.sig * noise
            self.var = self.sig ** 2
            self.regularization_loss = tf.reduce_mean(w ** 2 + b ** 2)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
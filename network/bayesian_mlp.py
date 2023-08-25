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

            w = mu_w + tf.exp(tf.clip_by_value(logsig_w, clip_min, clip_max)) * noise_w
            b = mu_b + tf.exp(tf.clip_by_value(logsig_b, clip_min, clip_max)) * noise_b

            out = (tf.matmul(input_tensor, w) + b) / input_dim 
            if input_dropout != None:
                out = tf.nn.dropout(out, rate=input_dropout)

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            
            self.layer_output = out
            self.regularization_loss = tf.reduce_mean(mu_w ** 2 + logsig_w ** 2) + tf.reduce_mean(mu_b ** 2 + logsig_b ** 2)
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
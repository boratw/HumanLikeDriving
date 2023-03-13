import tensorflow.compat.v1 as tf
import math


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dims, hidden_nonlns=tf.nn.relu, output_nonln=None,
                    reuse=False, input_tensor=None, name=None):
        if name == None:
            name = "MLP"
        else:
            name = "MLP_" + name
        if hasattr(hidden_nonlns, "__iter__") == False and hidden_dims > 1:
            hidden_nonlns = [hidden_nonlns] * hidden_dims
        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor

            idim = input_dim
            out = self.layer_input
            for i, dim in enumerate(hidden_dims):
                w = tf.get_variable("w" + str(i), shape=[idim, dim], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (idim + dim)), math.sqrt(6.0 / (idim + dim)), dtype=tf.float32),
                    trainable=True)
                b = tf.get_variable("b" + str(i), shape=[idim, dim], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                out = tf.matmul(out, w) + b
                if i != hidden_dims - 1:
                    if hidden_nonlns[i] == tf.nn.leaky_relu:
                        out = tf.nn.leaky_relu(out, alpha=0.05)
                    elif hidden_nonlns[i] != None:
                        out = hidden_nonlns[i](out)
                    
                idim = dim

            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.05)
            elif output_nonln != None:
                out = hidden_nonlns[i](out)

            self.layer_output = out
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

    def build_add_weighted(self, source, weight):
        return [ tf.assign(target, (1 - weight) * target + weight * source) for target, source in zip(self.trainable_params, source.trainable_params)]

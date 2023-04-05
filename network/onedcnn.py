import tensorflow.compat.v1 as tf
import math
from network.mlp import MLP

class OneDCnn:
    def __init__(self, input_dim, output_dim, cnn_hidden_dims, fc_hidden_dims, hidden_nonlns=tf.nn.relu, 
                 reuse=False, input_tensor=None, name=None):
        if name == None:
            name = "OneDCnn"
        else:
            name = "OneDCnn_" + name
        if hasattr(hidden_nonlns, "__iter__") == False:
            hidden_nonlns = [hidden_nonlns] * (len(cnn_hidden_dims) + len(fc_hidden_dims))

        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor
            idim = input_dim
            out = self.layer_input
            for i, dim in enumerate(cnn_hidden_dims):
                w = tf.get_variable("wc" + str(i), shape=[1, 3, idim, dim], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (3 + idim + dim)), math.sqrt(6.0 / (3 + idim + dim)), dtype=tf.float32),
                    trainable=True)
                b = tf.get_variable("bc" + str(i), shape=[dim], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "VALID") + b
                out = tf.nn.max_pool2d(out, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")
                if hidden_nonlns[i] == tf.nn.leaky_relu:
                    out = tf.nn.leaky_relu(out, alpha=0.05)
                elif hidden_nonlns[i] != None:
                    out = hidden_nonlns[i](out)
                idim = dim
            out = tf.layers.Flatten()(out)
            idim = out.shape[1]

            for i, dim in enumerate(fc_hidden_dims):
                w = tf.get_variable("wf" + str(i), shape=[idim, dim], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (idim + dim)), math.sqrt(6.0 / (idim + dim)), dtype=tf.float32),
                    trainable=True)
                b = tf.get_variable("bf" + str(i), shape=[dim], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                out = tf.matmul(out, w) + b
                if hidden_nonlns[len(cnn_hidden_dims) + i] == tf.nn.leaky_relu:
                    out = tf.nn.leaky_relu(out, alpha=0.05)
                elif hidden_nonlns[len(cnn_hidden_dims) + i] != None:
                    out = hidden_nonlns[len(cnn_hidden_dims) + i](out)
                idim = dim

            w = tf.get_variable("wf" + str(len(fc_hidden_dims)), shape=[idim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (3 + idim + dim)), math.sqrt(6.0 / (3 + idim + dim)), dtype=tf.float32),
                trainable=True)
            b = tf.get_variable("bf" + str(len(fc_hidden_dims)), shape=[output_dim], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            out = tf.matmul(out, w) + b

            self.layer_output = out
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

    def build_add_weighted(self, source, weight):
        return [ tf.assign(target, (1 - weight) * target + weight * source) for target, source in zip(self.trainable_params, source.trainable_params)]

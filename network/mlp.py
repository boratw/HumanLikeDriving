import tensorflow.compat.v1 as tf
import math


class MLP:
    def __init__(self, input_dim, output_dim, hidden_dims, hidden_nonlns=tf.nn.relu, output_nonln=None,
                    reuse=False, input_tensor=None, use_dropout=False, input_dropout= None, name=None):
        if name == None:
            name = "MLP"
        else:
            name = "MLP_" + name
        if hasattr(hidden_nonlns, "__iter__") == False:
            hidden_nonlns = [hidden_nonlns] * len(hidden_dims)
        
        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor
            if use_dropout:
                if input_dropout is None:
                    self.layer_dropout = tf.placeholder(tf.float32, None)
                else:
                    self.layer_dropout = input_dropout

            idim = input_dim
            out = self.layer_input
            l2_loss = []
            for i, dim in enumerate(hidden_dims):
                w = tf.get_variable("w" + str(i), shape=[idim, dim], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(idim + dim), 1.0 / math.sqrt(idim + dim), dtype=tf.float32),
                    trainable=True)
                b = tf.get_variable("b" + str(i), shape=[dim], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                out = tf.matmul(out, w) + b
                l2_loss.append(tf.reduce_mean(out ** 2))
                if hidden_nonlns[i] == tf.nn.leaky_relu:
                    out = tf.nn.leaky_relu(out, alpha=0.001)
                elif hidden_nonlns[i] != None:
                    out = hidden_nonlns[i](out)
                if use_dropout:
                    out = tf.nn.dropout(out, rate=self.layer_dropout)
                idim = dim

            w = tf.get_variable("w" + str(len(hidden_dims)), shape=[idim, output_dim], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(idim + output_dim), 1.0 / math.sqrt(idim + output_dim), dtype=tf.float32),
                trainable=True)
            b = tf.get_variable("b" + str(len(hidden_dims)), shape=[output_dim], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)
            out = tf.matmul(out, w) + b
            l2_loss.append(tf.reduce_mean(out ** 2))
            if output_nonln == tf.nn.leaky_relu:
                out = tf.nn.leaky_relu(out, alpha=0.001)
            elif output_nonln != None:
                out = output_nonln(out)

            self.layer_output = out
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            self.l2_loss = tf.reduce_mean(l2_loss)

    def build_add_weighted(self, source, weight):
        return [ tf.assign(target, (1 - weight) * target + weight * source) for target, source in zip(self.trainable_params, source.trainable_params)]

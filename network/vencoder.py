import tensorflow.compat.v1 as tf
import math
from network.mlp import MLP

class VEncoder:
    def __init__(self, input_dim, latent_dim, hidden_dims, hidden_nonlns=tf.nn.relu,
                    additional_dim=None, additional_tensor=None, reuse=False, input_tensor=None, 
                    name=None):
        if name == None:
            name = "VEncoder"
        else:
            name = "VEncoder" + name

        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor

            if additional_dim != None:
                if additional_tensor is None:
                    self.layer_additional = tf.placeholder(tf.float32, [None, additional_dim])
                else:
                    self.layer_additional = additional_tensor
                layer_encoder_input = tf.concat([self.layer_input, self.layer_additional], axis=1)
                encoder_input_dim = input_dim + additional_dim
            else:
                layer_encoder_input = self.layer_input
                encoder_input_dim = input_dim

            self.layer_latent = tf.placeholder(tf.float32, [None, latent_dim])
        
            self.encoder = MLP(encoder_input_dim, latent_dim * 2, hidden_dims, hidden_nonlns, reuse=reuse,
                               input_tensor=layer_encoder_input, use_dropout=False, name="Enc")
            
            self.mu, self.logsig = tf.split(self.encoder.layer_output, [latent_dim, latent_dim], 1)
            self.logsig = tf.clip_by_value(self.logsig, -10, 2)
            self.sig = tf.exp(self.logsig)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=self.sig)
            self.layer_output = self.dist.sample()
            self.prior = tf.distributions.Normal(loc=tf.zeros_like(self.mu), scale=tf.ones_like(self.logsig))
            
            self.regularization_loss = tf.reduce_mean(self.dist.kl_divergence(self.prior))
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

import tensorflow.compat.v1 as tf
import math
from network.mlp import MLP

class VAE:
    def __init__(self, input_dim, latent_dim, hidden_dims, hidden_nonlns=tf.nn.relu,
                    additional_dim=None, additional_tensor=None, reuse=False, input_tensor=None, 
                    name=None):
        if name == None:
            name = "VAE"
        else:
            name = "VAE_" + name

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
        
            self.encoder = MLP(encoder_input_dim, latent_dim * 2, hidden_dims, hidden_nonlns, reuse=reuse,
                               input_tensor=layer_encoder_input, use_dropout=False, name="Enc")
            
            self.mu, self.logsig = tf.split(self.encoder.layer_output, [latent_dim, latent_dim], 1)
            self.logsig = tf.clip_by_value(self.logsig, -8, 2)
            self.sig = tf.exp(self.logsig)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=self.sig)
            self.x = self.dist.sample()
            self.prior = tf.distributions.Normal(loc=tf.zeros_like(self.mu), scale=tf.ones_like(self.logsig))

            if additional_dim != None:
                layer_decoder_input = tf.concat([self.x, self.layer_additional], axis=1)
                decoder_input_dim = latent_dim + additional_dim
            else:
                layer_decoder_input = self.x
                decoder_input_dim = latent_dim

            self.decoder = MLP(decoder_input_dim, input_dim, hidden_dims[::-1], hidden_nonlns, reuse=reuse,
                               input_tensor=layer_decoder_input, use_dropout=False, name="Dec")
            
            self.regularization_loss = tf.reduce_mean(self.dist.kl_divergence(self.prior))
            self.layer_output = self.decoder.layer_output


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

class VAE_Encoder:
    def __init__(self, input_dim, latent_dim, hidden_dims, hidden_nonlns=tf.nn.relu,
                    additional_dim=None, additional_tensor=None, reuse=False, input_tensor=None, 
                    use_dropout=False, input_dropout= None, name=None):
        if name == None:
            name = "VAE"
        else:
            name = "VAE_" + name

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
        
            self.encoder = MLP(encoder_input_dim, latent_dim * 2, hidden_dims, hidden_nonlns, reuse=reuse,
                               input_tensor=layer_encoder_input, use_dropout=use_dropout,
                               input_dropout=input_dropout,name="Enc")
            
            self.mu, self.logsig = tf.split(self.encoder.layer_output, [latent_dim, latent_dim], 1)
            self.logsig = tf.clip_by_value(self.logsig, -8, 2)
            self.sig = tf.exp(self.logsig)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=self.sig)
            self.layer_output = self.dist.sample()
            self.prior = tf.distributions.Normal(loc=tf.zeros_like(self.mu), scale=tf.ones_like(self.logsig))

            self.log_pi = self.dist.log_prob(self.layer_output)
            self.regularization_loss = tf.reduce_mean(self.dist.kl_divergence(self.prior))
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

class VAE_Decoder:
    def __init__(self, input_dim, latent_dim, hidden_dims, hidden_nonlns=tf.nn.relu,
                    additional_dim=None, additional_tensor=None, reuse=False, latent_tensor=None, 
                    name=None):
        if name == None:
            name = "VAE"
        else:
            name = "VAE_" + name

        with tf.variable_scope(name, reuse=reuse):
            if latent_tensor is None:
                dist = tf.distributions.Normal(loc=tf.zeros([1, latent_dim]), scale=tf.ones([1, latent_dim]))
                self.latent_input = dist.sample()
            else:
                self.latent_input = latent_tensor

            if additional_dim != None:
                if additional_tensor is None:
                    self.layer_additional = tf.placeholder(tf.float32, [None, additional_dim])
                else:
                    self.layer_additional = additional_tensor
                layer_decoder_input = tf.concat([self.latent_input, self.layer_additional], axis=1)
                decoder_input_dim = latent_dim + additional_dim
            else:
                layer_decoder_input = self.latent_input
                decoder_input_dim = latent_dim

            self.decoder = MLP(decoder_input_dim, input_dim, hidden_dims, hidden_nonlns, reuse=reuse,
                               input_tensor=layer_decoder_input, use_dropout=False, name="Dec")
            
            self.layer_output = self.decoder.layer_output


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)


class VAE_Reverse:
    def __init__(self, input_dim, latent_dim, hidden_dims, hidden_nonlns=tf.nn.relu,
                    additional_dim=None, additional_tensor=None, reuse=False, input_tensor=None, 
                    name=None):
        if name == None:
            name = "VAE"
        else:
            name = "VAE_" + name

        with tf.variable_scope(name, reuse=reuse):
            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, latent_dim])
            else:
                self.layer_input = input_tensor

            if additional_dim != None:
                if additional_tensor is None:
                    self.layer_additional = tf.placeholder(tf.float32, [None, additional_dim])
                else:
                    self.layer_additional = additional_tensor
                layer_decoder_input = tf.concat([self.layer_input, self.layer_additional], axis=1)
                decoder_input_dim = latent_dim + additional_dim
            else:
                layer_decoder_input = self.layer_input
                decoder_input_dim = latent_dim

            self.decoder = MLP(decoder_input_dim, input_dim, hidden_dims[::-1], hidden_nonlns, reuse=reuse,
                               input_tensor=layer_decoder_input, use_dropout=False, name="Dec")
            

            if additional_dim != None:
                layer_encoder_input = tf.concat([self.decoder.layer_output, self.layer_additional], axis=1)
                encoder_input_dim = input_dim + additional_dim
            else:
                layer_encoder_input = self.decoder.layer_output
                encoder_input_dim = input_dim
        
            self.encoder = MLP(encoder_input_dim, latent_dim * 2, hidden_dims, hidden_nonlns, reuse=reuse,
                               input_tensor=layer_encoder_input, use_dropout=False, name="Enc")
            
            self.mu, self.logsig = tf.split(self.encoder.layer_output, [latent_dim, latent_dim], 1)
            self.logsig = tf.clip_by_value(self.logsig, -8, 2)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=tf.exp(self.logsig))
            self.x = self.dist.sample()
            self.prior = tf.distributions.Normal(loc=tf.zeros_like(self.mu), scale=tf.ones_like(self.logsig))


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
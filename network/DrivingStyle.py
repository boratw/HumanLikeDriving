import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn

def fully_connect(name, l, idim, odim):
    w = tf.get_variable(name + "_w", shape=[idim, odim], dtype=tf.float32, 
        initializer=tf.random_uniform_initializer(-1.0 / math.sqrt(idim + odim), 1.0 / math.sqrt(idim + odim), dtype=tf.float32), trainable=True)
    b = tf.get_variable(name + "_b", shape=[odim], dtype=tf.float32, 
        initializer=tf.zeros_initializer(dtype=tf.float32), trainable=True)
    return tf.matmul(l, w) + b



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, global_latent_len = 4,
                 locally_encoded_state_length = 16, decoder_shuffle_num=8, shuffle_loss_ratio=0.5,
                 locally_encoder_lr = 0.0001, global_encoder_lr=0.0001, global_regularizer_weight= 0.001):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        self.route_len = route_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_route = tf.placeholder(tf.float32, [None, action_len, route_len])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)
            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])

            self.locally_encoder_input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            self.locally_encoder_h = fully_connect("le1", self.layer_input_state, state_len + action_len * route_len, 256)
            self.locally_encoder_h = tf.nn.leaky_relu(self.locally_encoder_h, alpha=0.001)
            self.locally_encoded_state = fully_connect("le2", self.locally_encoder_h, 256, locally_encoded_state_length)
            self.locally_encoded_state = tf.nn.tanh(self.locally_encoded_state)
            self.locally_encoder_output = fully_connect("le3", self.locally_encoded_state, locally_encoded_state_length, nextstate_len)

            self.locally_encoder_error = self.locally_encoder_output - self.layer_input_nextstate
            self.global_encoder_input = tf.concat([self.locally_encoded_state, self.locally_encoder_error], axis=1)
            self.global_encoder = VAE_Encoder(locally_encoded_state_length + nextstate_len, global_latent_len, [128, 128], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder", reuse=True, use_dropout=True, input_dropout=self.layer_input_dropout )
            
            self.global_decoder_input = tf.concat([self.locally_encoded_state, self.global_encoder.layer_output], axis=1)
            self.global_decoder = MLP(global_latent_len + locally_encoded_state_length, nextstate_len, [128, 128], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_decoder_input, name="GlobalDecoder", use_dropout=True, input_dropout=self.layer_input_dropout  )
            
            shuffled_global_latent = tf.tile(self.global_encoder.layer_output, [decoder_shuffle_num, 1])
            locally_encoded_state_batch = tf.tile(self.locally_encoded_state, [decoder_shuffle_num, 1])
            locally_encoder_error_batch = tf.tile(self.global_encoder.layer_output, [decoder_shuffle_num, 1])
            tf.random.shuffle(shuffled_global_latent)
            self.shuffled_global_decoder_input = tf.concat([locally_encoded_state_batch, tf.stop_gradient(shuffled_global_latent)], axis=1)
            self.shuffled_global_decoder = MLP(global_latent_len + locally_encoded_state_length, nextstate_len, [128, 128], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_decoder_input, name="GlobalDecoder", use_dropout=True, input_dropout=self.layer_input_dropout, reuse=True  )


            self.locally_encoder_loss = tf.reduce_mean((self.locally_encoder_output - self.layer_input_nextstate) ** 2, axis=0)
            self.encoder_reconstruction_loss = tf.reduce_mean((self.global_decoder.layer_output - self.locally_encoder_error), axis=0)
            self.shuffle_encoder_reconstruction_loss = tf.reduce_mean((self.shuffled_global_decoder.layer_output - locally_encoder_error_batch), axis=0)

            self.global_encoder_loss = tf.reduce_mean(self.encoder_reconstruction_loss + self.shuffle_encoder_reconstruction_loss * shuffle_loss_ratio) \
                + self.global_encoder.regularization_loss * global_regularizer_weight

            self.locally_encoder_optimizer = tf.train.AdamOptimizer(locally_encoder_lr)
            self.locally_encoder_train_action = self.locally_encoder_optimizer.minimize(loss = self.locally_encoder_loss)
            self.global_encoder_optimizer = tf.train.AdamOptimizer(global_encoder_lr)
            self.global_encoder_train_action = self.global_encoder_optimizer.minimize(loss = self.global_encoder_loss)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_local_encoder_loss = np.array([0.] * self.nextstate_len)
        self.log_global_encoder_loss = np.array([0.] * self.nextstate_len)
        self.log_global_shuffle_loss = np.array([0.] * self.nextstate_len)
        self.log_global_encoder_kl = 0.
        self.log_num = 0

    def network_update(self):
        self.log_local_encoder_loss = np.array([0.] * self.nextstate_len)
        self.log_global_encoder_loss = np.array([0.] * self.nextstate_len)
        self.log_global_shuffle_loss = np.array([0.] * self.nextstate_len)
        self.log_global_encoder_kl = 0.
        self.log_num = 0
            
    def optimize(self, input_state, input_nextstate, input_route):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route}
        sess = tf.get_default_session()
        _, l1 = sess.run([self.locally_encoder_train_action, self.locally_encoder_loss] ,input_list)
        _, l2, l3, l4 = sess.run([self.global_encoder_train_action, self.encoder_reconstruction_loss, 
                                  self.shuffle_encoder_reconstruction_loss, self.global_encoder.regularization_loss],input_list)
        
        self.log_local_encoder_loss += l1
        self.log_global_encoder_loss += l2
        self.log_global_shuffle_loss += l3
        self.log_global_encoder_kl += l4
        self.log_num += 1
       
    def log_caption(self):
        return "\t" + self.name + "_LocallyReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_GlobalReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_GlobalShuffleLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_GlobalKLReg\t"  

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_local_encoder_loss[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_global_encoder_loss[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_global_shuffle_loss[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_global_encoder_kl / log_num)
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tLocallyReconLoss       : " + " ".join([str(self.log_local_encoder_loss[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tGlobalReconLoss        : " + " ".join([str(self.log_global_encoder_loss[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tGlobalShuffleLoss      : " + " ".join([str(self.log_global_shuffle_loss[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tGlobalKLReg            : " + str(self.log_global_encoder_kl / log_num) )
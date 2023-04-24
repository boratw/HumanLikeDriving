import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, agent_for_each_train=16, state_len = 59, global_latent_len = 4, 
                 teacher_learner_lr = 0.001, global_learner_lr = 0.001, global_regularizer_weight= 0.001):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, 2])


            self.teacher = MLP(state_len,  2, [512, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state)


            self.global_encoder_input = tf.concat([(self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output)) * 10., self.layer_input_state], axis=1)
            self.global_encoder = MLP(2 + state_len, global_latent_len, [128, 128], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder" )
            global_latent = tf.reshape(self.global_encoder.layer_output, [agent_for_each_train, -1, global_latent_len])
            global_latent = tf.reduce_sum(tf.math.exp(tf.clip_by_value(global_latent, -100, 4)), axis=1)
            self.global_latent = global_latent / tf.stop_gradient(tf.reduce_sum(global_latent, axis=1, keep_dims=True))
            global_latent_batch = tf.tile(global_latent, [1, tf.shape(self.global_encoder.layer_output)[0] // agent_for_each_train])
            self.global_latent_batch = tf.reshape(global_latent_batch, [-1, global_latent_len])


            self.global_encoder_input = tf.concat([self.global_latent_batch, self.layer_input_state], axis=1)
            self.global_decoder = MLP(global_latent_len + state_len, 2, [128, 128], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_encoder_input, name="GlobalDecoder" )

            self.teacher_loss = tf.reduce_mean((self.layer_input_nextstate - self.teacher.layer_output) ** 2)
            self.global_reconstruction_loss = tf.reduce_mean((self.layer_input_nextstate - self.global_decoder.layer_output) ** 2)
            self.global_regularization_loss = tf.reduce_mean(self.global_encoder.layer_output ** 2)
            self.global_loss = self.global_reconstruction_loss + self.global_regularization_loss * global_regularizer_weight

            self.global_mu_var = tf.math.reduce_variance(self.global_latent, axis=0)

            self.teacher_optimizer = tf.train.AdamOptimizer(teacher_learner_lr)
            self.teacher_train_action = self.teacher_optimizer.minimize(loss = self.teacher_loss, var_list=self.teacher.trainable_params)
            self.global_optimizer = tf.train.AdamOptimizer(global_learner_lr)
            self.global_train_action = self.global_optimizer.minimize(loss = self.global_loss, var_list=[*self.global_encoder.trainable_params, *self.global_decoder.trainable_params] )

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_teacher_loss = 0.
        self.log_global_loss_rec = 0.
        self.log_global_loss_reg = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_num = 0

    def network_update(self):
        self.log_teacher_loss = 0.
        self.log_global_loss_rec = 0.
        self.log_global_loss_reg = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_num = 0
            
    def optimize(self, input_state, input_nextstate):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate}
        sess = tf.get_default_session()
        _, l1 = sess.run([self.teacher_train_action, self.teacher_loss] ,input_list)
        _, l2, l3, l4 = sess.run([self.global_train_action, self.global_reconstruction_loss, self.global_regularization_loss, self.global_mu_var],input_list)
        
        self.log_teacher_loss += l1
        self.log_global_loss_rec += l2
        self.log_global_loss_reg += l3
        self.log_global_muvar += l4
        self.log_num += 1

       
    def log_caption(self):
        return "\t" + self.name + "_TeacherLoss\t"  + self.name + "_GlobalReconLoss\t" + self.name + "_GlobalRegLoss\t" \
            + self.name + "_GlobalMuvar\t" + "\t".join([ "" for _ in range(self.global_latent_len)])  

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + str(self.log_teacher_loss / log_num) + "\t" + str(self.log_global_loss_rec / log_num) + "\t" + str(self.log_global_loss_reg / log_num) + "\t"\
            + "\t".join([str(self.log_global_muvar[i] / log_num) for i in range(self.global_latent_len)]) 
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tTeacherLoss          : " + str(self.log_teacher_loss / log_num) \
            + "\n\tGlobalReconLoss      : " + str(self.log_global_loss_rec / log_num) \
            + "\n\tGlobalRegLoss        : " + str(self.log_global_loss_reg / log_num) \
            + "\n\tGlobalMuvar          : " + " ".join([str(self.log_global_muvar[i] / log_num)[:8] for i in range(self.global_latent_len)]) )
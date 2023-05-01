import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, agent_for_each_train=16, state_len = 59, global_latent_len = 4, 
                 teacher_learner_lr = 0.001, global_learner_lr_start = 0.001, global_learner_lr_end = 0.0001, global_learner_lr_step = 1000, global_regularizer_weight= 0.001):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, 2])
            self.layer_input_global_latent = tf.placeholder(tf.float32, [None, global_latent_len])
            self.layer_iteration_num = tf.placeholder(tf.int32, None)
            self.global_lr = global_learner_lr_end + tf.exp(-self.layer_iteration_num / global_learner_lr_step) * (global_learner_lr_start - global_learner_lr_end)

            #self.teacher = MLP(state_len,  2, [512, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state, name="Teacher")


            #self.global_encoder_input = tf.concat([self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output), self.layer_input_state], axis=1)
            self.global_encoder_input = tf.concat([self.layer_input_nextstate, self.layer_input_state], axis=1)
            self.global_encoder = MLP(2 + state_len, global_latent_len, [256, 256], hidden_nonlns = tf.nn.tanh, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder" )
            global_latent = tf.reshape(self.global_encoder.layer_output, [agent_for_each_train, -1, global_latent_len])
            global_latent = tf.reduce_sum(global_latent, axis=1, keep_dims=True)
            self.global_latent = global_latent / tf.stop_gradient(tf.math.sqrt(tf.reduce_sum(global_latent ** 2, axis=2, keep_dims=True)))
            #self.global_latent = tf.reduce_sum(tf.math.exp(tf.clip_by_value(global_latent, -100, 4)), axis=1, keep_dims=True)
            global_latent_batch = tf.tile(self.global_latent, [1, tf.shape(self.global_encoder.layer_output)[0] // agent_for_each_train, 1])
            self.global_latent_batch = tf.reshape(global_latent_batch, [-1, global_latent_len])
            self.global_latent_output = self.global_encoder.layer_output

            self.global_decoder_input = tf.concat([self.global_latent_batch, self.layer_input_state], axis=1)
            self.global_decoder = MLP(global_latent_len + state_len, 2, [256, 256], hidden_nonlns = tf.nn.tanh, 
                        input_tensor=self.global_decoder_input, name="GlobalDecoder" )

            global_latent_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_state], axis=1)
            self.global_latent_decoder = MLP(global_latent_len + state_len, 2, [256, 256], hidden_nonlns = tf.nn.tanh, 
                        input_tensor=global_latent_decoder_input, name="GlobalDecoder", reuse=True )
            #self.global_latent_decoder_output = self.teacher.layer_output + self.global_decoder.layer_output
            self.global_latent_decoder_output = self.global_latent_decoder.layer_output
            
            #self.teacher_loss = tf.reduce_mean((self.layer_input_nextstate - self.teacher.layer_output) ** 2)
            #self.global_reconstruction_loss = tf.reduce_mean(((self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output)) - self.global_decoder.layer_output) ** 2)
            self.global_reconstruction_loss = tf.reduce_mean((self.layer_input_nextstate  - self.global_decoder.layer_output) ** 2)
            self.global_regularization_loss = tf.reduce_mean(self.global_encoder.layer_output ** 2)
            self.global_loss = self.global_reconstruction_loss + self.global_regularization_loss * global_regularizer_weight

            self.global_mu_var = tf.math.reduce_variance(self.global_latent, axis=[0, 1])

            #self.teacher_optimizer = tf.train.AdamOptimizer(teacher_learner_lr)
            #self.teacher_train_action = self.teacher_optimizer.minimize(loss = self.teacher_loss, var_list=self.teacher.trainable_params)
            self.global_optimizer = tf.train.AdamOptimizer(self.global_lr)
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
            
    def optimize(self, iteration_num, input_state, input_nextstate):
        input_list = {self.layer_iteration_num : iteration_num, self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate}
        sess = tf.get_default_session()
        #_, l1 = sess.run([self.teacher_train_action, self.teacher_loss] ,input_list)
        _, l2, l3, l4 = sess.run([self.global_train_action, self.global_reconstruction_loss, self.global_regularization_loss, self.global_mu_var],input_list)
        
        #self.log_teacher_loss += l1
        self.log_global_loss_rec += l2
        self.log_global_loss_reg += l3
        self.log_global_muvar += l4
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate}
        sess = tf.get_default_session()
        l1 = sess.run(self.global_latent_output, input_list)
        return l1


    def get_global_decoded(self, input_state, input_nextstate, input_global_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_global_latent : input_global_latent}
        sess = tf.get_default_session()
        l1 = sess.run(self.global_latent_decoder_output, input_list)
        return l1

       
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
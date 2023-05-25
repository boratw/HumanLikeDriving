import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, agent_for_each_train=16, state_len = 59, nextstate_len=2, global_latent_len = 4, local_latent_len = 1,
                 learner_lr_start = 0.001, learner_lr_end = 0.0001, learner_lr_step = 1000, global_regularizer_weight= 0.01, local_regularizer_weight=0.01,
                 l2_regularizer_weight=0.001):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        self.local_latent_len = local_latent_len
        self.nextstate_len = nextstate_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_global_latent = tf.placeholder(tf.float32, [None, global_latent_len])
            if local_latent_len != 0:
                self.layer_input_local_latent = tf.placeholder(tf.float32, [None, local_latent_len])
            self.layer_iteration_num = tf.placeholder(tf.int32, None)
            self.layer_dropout = tf.placeholder(tf.float32, None)
            self.lr = learner_lr_end + tf.exp(-self.layer_iteration_num / learner_lr_step) * (learner_lr_start - learner_lr_end)

            #self.teacher = MLP(state_len,  2, [512, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state, name="Teacher")


            #self.global_encoder_input = tf.concat([self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output), self.layer_input_state], axis=1)
            self.global_encoder_input = tf.concat([self.layer_input_nextstate, self.layer_input_state], axis=1)
            self.global_encoder = MLP(nextstate_len + state_len, global_latent_len, [256, 256, 256], hidden_nonlns = tf.nn.elu, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder", use_dropout=True, input_dropout=self.layer_dropout )
            global_latent = tf.reshape(self.global_encoder.layer_output, [agent_for_each_train, -1, global_latent_len])
            global_latent = tf.reduce_sum(global_latent, axis=1, keep_dims=True)
            self.global_latent = global_latent / tf.stop_gradient(tf.math.sqrt(tf.reduce_sum(global_latent ** 2, axis=2, keep_dims=True)) + 1e-7)
            #self.global_latent = tf.reduce_sum(tf.math.exp(tf.clip_by_value(global_latent, -100, 4)), axis=1, keep_dims=True)
            global_latent_batch = tf.tile(self.global_latent, [1, 1, tf.shape(self.global_encoder.layer_output)[0] // agent_for_each_train])
            self.global_latent_batch = tf.reshape(global_latent_batch, [-1, global_latent_len])
            self.global_latent_output = self.global_encoder.layer_output

            if local_latent_len != 0:
                self.local_encoder = MLP(nextstate_len + state_len, local_latent_len, [256, 256, 128], hidden_nonlns = tf.nn.elu, 
                            input_tensor=self.global_encoder_input, name="LocalEncoder", use_dropout=True, input_dropout=self.layer_dropout )
                self.decoder_input = tf.concat([self.global_latent_batch, self.local_encoder.layer_output, self.layer_input_state], axis=1)
                latent_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_local_latent, self.layer_input_state], axis=1)
            else:
                self.decoder_input = tf.concat([self.global_latent_batch, self.layer_input_state], axis=1)
                latent_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_state], axis=1)

            

            self.decoder = MLP(global_latent_len + local_latent_len + state_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.elu, 
                        input_tensor=self.decoder_input, name="Decoder", use_dropout=True, input_dropout=self.layer_dropout  )

            self.latent_decoder = MLP(global_latent_len + local_latent_len + state_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.elu, 
                        input_tensor=latent_decoder_input, name="Decoder", reuse=True, use_dropout=True, input_dropout=self.layer_dropout )
            self.latent_decoder_output = self.latent_decoder.layer_output
            
            #self.teacher_loss = tf.reduce_mean((self.layer_input_nextstate - self.teacher.layer_output) ** 2)
            #self.global_reconstruction_loss = tf.reduce_mean(((self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output)) - self.global_decoder.layer_output) ** 2)
            self.reconstruction_loss = tf.reduce_mean(tf.abs(self.layer_input_nextstate  - self.decoder.layer_output), axis=0)
            self.global_regularization_loss = tf.reduce_mean(tf.reduce_mean(self.global_encoder.layer_output, axis=0) ** 2.) + (tf.reduce_mean(tf.math.reduce_variance(self.global_encoder.layer_output, axis=0)) - 1.) ** 2.
            self.global_l2_loss = self.global_encoder.l2_loss + self.decoder.l2_loss

            if local_latent_len != 0:
                self.local_regularization_loss = tf.reduce_mean(self.local_encoder.layer_output) ** 2. + (tf.reduce_mean(tf.math.reduce_variance(self.global_encoder.layer_output, axis=0)) - 1.) ** 2.
                self.loss = tf.reduce_mean(self.reconstruction_loss) + self.global_regularization_loss * global_regularizer_weight +  self.local_regularization_loss * local_regularizer_weight \
                    + self.global_l2_loss * l2_regularizer_weight
            else:
                self.loss = tf.reduce_mean(self.reconstruction_loss) + self.global_regularization_loss * global_regularizer_weight + self.global_l2_loss * l2_regularizer_weight

            self.global_mu_var = tf.math.reduce_variance(self.global_latent, axis=[0, 1])

            #self.teacher_optimizer = tf.train.AdamOptimizer(teacher_learner_lr)
            #self.teacher_train_action = self.teacher_optimizer.minimize(loss = self.teacher_loss, var_list=self.teacher.trainable_params)
            self.global_optimizer = tf.train.AdamOptimizer(self.lr)
            self.global_train_action = self.global_optimizer.minimize(loss = self.loss)
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_global_loss_reg = 0.
        self.log_global_loss_l2 = 0.
        self.log_local_loss_reg = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_num = 0

    def network_update(self):
        self.log_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_global_loss_reg = 0.
        self.log_global_loss_l2 = 0.
        self.log_local_loss_reg = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_num = 0
            
    def optimize(self, iteration_num, input_state, input_nextstate):
        input_list = {self.layer_iteration_num : iteration_num, self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate,
                      self.layer_dropout : 0.1 }
        sess = tf.get_default_session()
        #_, l1 = sess.run([self.teacher_train_action, self.teacher_loss] ,input_list)
        if self.local_latent_len == 0:
            _, l2, l3, l4, l6  = sess.run([self.global_train_action, self.reconstruction_loss, self.global_regularization_loss, self.global_l2_loss, self.global_mu_var],input_list)
            l5 = 0.
        else:
            _, l2, l3, l4, l5, l6 = sess.run([self.global_train_action, self.reconstruction_loss, self.global_regularization_loss, self.global_l2_loss, self.local_regularization_loss, self.global_mu_var],input_list)
        
        #self.log_teacher_loss += l1
        self.log_loss_rec += l2
        self.log_global_loss_reg += l3
        self.log_global_loss_l2 += l4
        self.log_local_loss_reg += l5
        self.log_global_muvar += l6
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        if self.local_latent_len == 0:
            l1 = sess.run(self.global_encoder.layer_output, input_list)
            l2 = np.zeros((l1.shape[0], 1))
        else:
            l1, l2 = sess.run([self.global_encoder.layer_output, self.local_encoder.layer_output] , input_list)
        return l1, l2


    def get_global_decoded(self, input_state, input_nextstate, input_global_latent, input_local_latent=None):
        if self.local_latent_len == 0:
            input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_global_latent : input_global_latent, self.layer_dropout : 0.0}
        else:
            input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_global_latent : input_global_latent, 
                          self.layer_input_local_latent : input_local_latent, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        l1 = sess.run(self.latent_decoder_output, input_list)
        return l1

       
    def log_caption(self):
        return "\t" + self.name + "_ReconLoss\t" + self.name + "_GlobalRegLoss\t" + self.name + "_GlobalL2Loss\t" + self.name + "_LocalRegLoss\t"  \
            + self.name + "_GlobalMuvar\t" + "\t".join([ "" for _ in range(self.global_latent_len)])  

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_loss_rec[i] / log_num) for i in range(self.nextstate_len)])  + "\t"\
            + str(self.log_global_loss_reg / log_num) + "\t" + str(self.log_global_loss_l2 / log_num) + "\t"\
            + "\t" + str(self.log_local_loss_reg / log_num) + "\t"\
            + "\t".join([str(self.log_global_muvar[i] / log_num) for i in range(self.global_latent_len)]) 
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tReconLoss            : " + " ".join([str(self.log_loss_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tGlobalRegLoss        : " + str(self.log_global_loss_reg / log_num) \
            + "\n\tGlobalL2Loss         : " + str(self.log_global_loss_l2 / log_num) \
            + "\n\tLocalRegLoss         : " + str(self.log_local_loss_reg / log_num) \
            + "\n\tGlobalMuvar          : " + " ".join([str(self.log_global_muvar[i] / log_num)[:8] for i in range(self.global_latent_len)]) )
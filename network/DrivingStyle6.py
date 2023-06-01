import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, agent_for_each_train=8, state_len = 59, nextstate_len=2, global_latent_len = 4,
                 learner_lr_start = 0.001, learner_lr_end = 0.0001, learner_lr_step = 1000, global_regularizer_weight= 0.01, 
                 l2_regularizer_weight=0.001, decoder_shuffle_latent=8, encoder_lr=1, decoder_lr=1.):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        self.nextstate_len = nextstate_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_global_latent = tf.placeholder(tf.float32, [None, global_latent_len])
            self.layer_iteration_num = tf.placeholder(tf.int32, None)
            self.layer_dropout = tf.placeholder(tf.float32, None)
            self.encoder_lr = encoder_lr * (learner_lr_end + tf.exp(-self.layer_iteration_num / learner_lr_step) * (learner_lr_start - learner_lr_end))
            self.decoder_lr = decoder_lr * (learner_lr_end + tf.exp(-self.layer_iteration_num / learner_lr_step) * (learner_lr_start - learner_lr_end))

            #self.teacher = MLP(state_len,  2, [512, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state, name="Teacher")


            #self.global_encoder_input = tf.concat([self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output), self.layer_input_state], axis=1)
            self.global_encoder_input = tf.concat([self.layer_input_nextstate, self.layer_input_state], axis=1)
            self.global_encoder = VAE_Encoder(nextstate_len + state_len, global_latent_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder")
            
            self.decoder_input = tf.concat([self.global_encoder.layer_output, self.layer_input_state], axis=1)
            self.decoder = MLP(global_latent_len + state_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.decoder_input, name="Decoder", use_dropout=False  )
            
            global_latent = tf.tile(self.global_encoder.layer_output, [decoder_shuffle_latent + 1, 1])
            global_latent_batch = tf.split(global_latent, agent_for_each_train * (decoder_shuffle_latent + 1), axis=0)
            for g in global_latent_batch[agent_for_each_train:]:
                tf.random.shuffle(g)
            self.global_latent_batch = tf.reshape(tf.stack(global_latent_batch, axis=0), [-1, global_latent_len])

            layer_input_state_batch = tf.tile(self.layer_input_state, [decoder_shuffle_latent + 1, 1])
            layer_input_nextstate_batch = tf.tile(self.layer_input_nextstate, [decoder_shuffle_latent + 1, 1])
            self.batched_decoder_input = tf.concat([self.global_latent_batch, layer_input_state_batch], axis=1)
            self.batched_decoder = MLP(global_latent_len + state_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.batched_decoder_input, name="Decoder", reuse=True, use_dropout=False  )
            
            latent_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_state], axis=1)
            self.latent_decoder = MLP(global_latent_len + state_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=latent_decoder_input, name="Decoder", reuse=True, use_dropout=False )
            self.latent_decoder_output = self.latent_decoder.layer_output
            
            #self.teacher_loss = tf.reduce_mean((self.layer_input_nextstate - self.teacher.layer_output) ** 2)
            #self.global_reconstruction_loss = tf.reduce_mean(((self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output)) - self.global_decoder.layer_output) ** 2)
            self.encoder_reconstruction_loss = tf.reduce_mean(tf.abs(self.layer_input_nextstate - self.decoder.layer_output), axis=0)
            self.encoder_kl_loss = self.global_encoder.regularization_loss
            self.encoder_l2_loss = self.global_encoder.encoder.l2_loss
            self.encoder_loss = tf.reduce_mean(self.encoder_reconstruction_loss) + self.encoder_kl_loss * global_regularizer_weight + self.encoder_l2_loss * l2_regularizer_weight

            reconstruction_loss_batch = tf.split(tf.abs(layer_input_nextstate_batch - self.batched_decoder.layer_output), decoder_shuffle_latent + 1, axis=0)
            self.decoder_reconstruction_loss = tf.reduce_mean(reconstruction_loss_batch[0], axis=0) + tf.reduce_mean(reconstruction_loss_batch[1:], axis=[0, 1])
            self.decoder_l2_loss = self.decoder.l2_loss
            self.decoder_loss = tf.reduce_mean(self.decoder_reconstruction_loss) + self.decoder_l2_loss * l2_regularizer_weight

            self.latent_mumean = tf.math.reduce_mean(self.global_encoder.mu, axis=0)
            self.latent_muvar = tf.math.reduce_variance(self.global_encoder.mu, axis=0)
            self.latent_logsig = tf.math.reduce_mean(self.global_encoder.logsig, axis=0)

            self.global_optimizer = tf.train.AdamOptimizer(self.encoder_lr)
            self.global_train_action = self.global_optimizer.minimize(loss = self.encoder_loss, var_list=self.global_encoder.trainable_params)
            self.decoder_optimizer = tf.train.AdamOptimizer(self.decoder_lr)
            self.decoder_train_action = self.decoder_optimizer.minimize(loss = self.decoder_loss, var_list=self.decoder.trainable_params)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_encoder_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_encoder_loss_kl = 0.
        self.log_encoder_loss_l2 = 0.
        self.log_decoder_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_decoder_loss_l2 = 0.
        self.log_latent_mumean = np.array([0.] * self.global_latent_len)
        self.log_latent_muvar = np.array([0.] * self.global_latent_len)
        self.log_latent_logsig = np.array([0.] * self.global_latent_len)
        self.log_num = 0

    def network_update(self):
        self.log_encoder_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_encoder_loss_kl = 0.
        self.log_encoder_loss_l2 = 0.
        self.log_decoder_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_decoder_loss_l2 = 0.
        self.log_latent_mumean = np.array([0.] * self.global_latent_len)
        self.log_latent_muvar = np.array([0.] * self.global_latent_len)
        self.log_latent_logsig = np.array([0.] * self.global_latent_len)
        self.log_num = 0
            
    def optimize(self, iteration_num, input_state, input_nextstate):
        input_list = {self.layer_iteration_num : iteration_num, self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate,
                      self.layer_dropout : 0.1 }
        sess = tf.get_default_session()
        _, l1, l2, l3, l6, l7, l8 = sess.run([self.global_train_action, self.encoder_reconstruction_loss, self.encoder_kl_loss, self.encoder_l2_loss,
                                              self.latent_mumean, self.latent_muvar, self.latent_logsig],input_list)
        _, l4, l5 = sess.run([self.decoder_train_action, self.decoder_reconstruction_loss, self.decoder_l2_loss],input_list)

        self.log_encoder_loss_rec += l1
        self.log_encoder_loss_kl += l2
        self.log_encoder_loss_l2 += l3
        self.log_decoder_loss_rec += l4
        self.log_decoder_loss_l2 += l5
        self.log_latent_mumean += l6
        self.log_latent_muvar += l7
        self.log_latent_logsig += l8
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.global_encoder.mu, self.global_encoder.sig], input_list)
        return l1, l2


    def get_global_decoded(self, input_state, input_nextstate, input_global_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_global_latent : input_global_latent, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        l1 = sess.run(self.latent_decoder_output, input_list)
        return l1

       
    def log_caption(self):
        return "\t" + self.name + "_GlobalRegLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)])  + self.name + "_GlobalKLLoss\t" + self.name + "_GlobalL2Loss\t" \
            + self.name + "_DecoderRegLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) + self.name + "_DecoderL2Loss" \
            + self.name + "_LatentMuMean\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_LatentMuVar\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_LatentLogSig\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) 

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_encoder_loss_rec[i] / log_num) for i in range(self.nextstate_len)])  + "\t"\
            + str(self.log_encoder_loss_kl / log_num) + "\t" + str(self.log_encoder_loss_l2 / log_num) + "\t"\
            + "\t".join([str(self.log_decoder_loss_rec[i] / log_num) for i in range(self.nextstate_len)])  + "\t"\
            + str(self.log_decoder_loss_l2 / log_num)+ "\t" \
            + "\t".join([str(self.log_latent_mumean[i] / log_num) for i in range(self.global_latent_len)])+ "\t" \
            + "\t".join([str(self.log_latent_muvar[i] / log_num) for i in range(self.global_latent_len)])+ "\t" \
            + "\t".join([str(self.log_latent_logsig[i] / log_num) for i in range(self.global_latent_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tGlobalReconLoss      : " + " ".join([str(self.log_encoder_loss_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tGlobalRegLoss        : " + str(self.log_encoder_loss_kl / log_num) \
            + "\n\tGlobalL2Loss         : " + str(self.log_encoder_loss_l2 / log_num) \
            + "\n\tDecoderReconLoss     : " + " ".join([str(self.log_decoder_loss_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tDecoderL2Loss        : " + str(self.log_decoder_loss_l2 / log_num) \
            + "\n\tLatentMuMean         : " + " ".join([str(self.log_latent_mumean[i] / log_num)[:8] for i in range(self.global_latent_len)])\
            + "\n\tLatentMuvar          : " + " ".join([str(self.log_latent_muvar[i] / log_num)[:8] for i in range(self.global_latent_len)])\
            + "\n\tLatentLogSig         : " + " ".join([str(self.log_latent_logsig[i] / log_num)[:8] for i in range(self.global_latent_len)]))
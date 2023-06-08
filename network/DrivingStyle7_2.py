import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, agent_for_each_train=8, state_len = 59, route_len = 10, nextstate_len=2, action_len=3, global_latent_len = 4,
                 learner_lr_start = 0.001, learner_lr_end = 0.0001, learner_lr_step = 1000, global_regularizer_weight= 0.01, latent_variance_weight = 0.1,
                 l2_regularizer_weight=0.001):
        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.global_latent_len = global_latent_len
        self.nextstate_len = nextstate_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_action = tf.placeholder(tf.int32, [None])
            self.layer_input_route = tf.placeholder(tf.float32, [None, action_len, route_len])
            self.layer_input_global_latent = tf.placeholder(tf.float32, [None, global_latent_len])
            self.layer_iteration_num = tf.placeholder(tf.int32, None)
            self.layer_dropout = tf.placeholder(tf.float32, None)
            self.lr = learner_lr_end + tf.exp(-self.layer_iteration_num / learner_lr_step) * (learner_lr_start - learner_lr_end)

            #self.teacher = MLP(state_len,  2, [512, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state, name="Teacher")


            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])
            layer_input_route_selected = tf.gather(self.layer_input_route, self.layer_input_action, axis=1, batch_dims=1)

            #self.global_encoder_input = tf.concat([self.layer_input_nextstate - tf.stop_gradient(self.teacher.layer_output), self.layer_input_state], axis=1)
            self.global_encoder_input = tf.concat([self.layer_input_nextstate, self.layer_input_state, layer_input_route_flatten], axis=1)
            self.global_encoder = VAE_Encoder(nextstate_len + state_len+ route_len * action_len, global_latent_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.global_encoder_input, name="GlobalEncoder")
            
            self.route_decoder_input = tf.concat([self.global_encoder.layer_output, self.layer_input_state, layer_input_route_selected], axis=1)
            self.action_decoder_input = tf.concat([self.global_encoder.layer_output, self.layer_input_state, layer_input_route_flatten], axis=1)
            self.route_decoder = MLP(global_latent_len + state_len + route_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.route_decoder_input, name="RouteDecoder", use_dropout=False  )
            self.action_decoder = MLP(global_latent_len + state_len + route_len * action_len, action_len, [256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=self.action_decoder_input, name="ActionDecoder", use_dropout=False  )
            self.action_decoder_output = tf.nn.softmax(self.action_decoder.layer_output)
            
            global_latent_batch = tf.reshape(self.global_encoder.layer_output, [agent_for_each_train, -1, global_latent_len])
            self.global_latent_similarity = tf.reduce_mean(tf.math.reduce_std(global_latent_batch, axis=1), axis=0 )
            self.global_latent_difference = tf.math.reduce_std(tf.reduce_mean(global_latent_batch, axis=1), axis=0 )
            
            latent_action_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_state, layer_input_route_flatten], axis=1)
            self.latent_action_decoder = MLP(global_latent_len + state_len + route_len * action_len, action_len, [256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=latent_action_decoder_input, name="ActionDecoder", reuse=True, use_dropout=False )
            self.latent_action = tf.math.argmax(self.latent_action_decoder.layer_output, axis=1)
            layer_input_route_latent_selected = tf.gather(self.layer_input_route, self.latent_action, axis=1, batch_dims=1)
            latent_route_decoder_input = tf.concat([self.layer_input_global_latent, self.layer_input_state, layer_input_route_latent_selected], axis=1)
            self.latent_route_decoder = MLP(global_latent_len + state_len + route_len, nextstate_len, [256, 256, 256], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=latent_route_decoder_input, name="RouteDecoder", reuse=True, use_dropout=False )
            self.latent_route_decoder_output = self.latent_route_decoder.layer_output

            
            self.encoder_kl_loss = self.global_encoder.regularization_loss
            self.encoder_l2_loss = self.global_encoder.encoder.l2_loss

            self.route_decoder_reconstruction_loss = tf.reduce_mean(tf.abs(self.layer_input_nextstate - self.route_decoder.layer_output), axis=0)
            self.route_decoder_l2_loss = self.route_decoder.l2_loss
            self.route_decoder_loss = tf.reduce_mean(self.route_decoder_reconstruction_loss) + self.route_decoder_l2_loss * l2_regularizer_weight

            self.action_decoder_reconstruction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tf.one_hot(self.layer_input_action, action_len), self.action_decoder.layer_output))
            self.action_decoder_l2_loss = self.action_decoder.l2_loss
            self.action_decoder_loss = tf.reduce_mean(self.action_decoder_reconstruction_loss) + self.action_decoder_l2_loss * l2_regularizer_weight

            self.loss = (tf.reduce_mean(self.global_latent_similarity) - tf.reduce_mean(self.global_latent_difference) * 0.5) * latent_variance_weight \
                + self.encoder_kl_loss * global_regularizer_weight + self.encoder_l2_loss * l2_regularizer_weight \
                + tf.reduce_mean(self.route_decoder_reconstruction_loss) + self.route_decoder_l2_loss * l2_regularizer_weight \
                + tf.reduce_mean(self.action_decoder_reconstruction_loss) + self.action_decoder_l2_loss * l2_regularizer_weight

            self.latent_mumean = tf.math.reduce_mean(self.global_encoder.mu, axis=0)
            self.latent_muvar = tf.math.reduce_variance(self.global_encoder.mu, axis=0)
            self.latent_logsig = tf.math.reduce_mean(self.global_encoder.logsig, axis=0)

            self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_action = self.optimizer.minimize(loss = self.loss)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_encoder_loss_sim = np.array([0.] * self.global_latent_len)
        self.log_encoder_loss_dis = np.array([0.] * self.global_latent_len)
        self.log_encoder_loss_kl = 0.
        self.log_encoder_loss_l2 = 0.
        self.log_route_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_route_loss_l2 = 0.
        self.log_action_loss_rec = 0.
        self.log_action_loss_l2 = 0.
        self.log_latent_mumean = np.array([0.] * self.global_latent_len)
        self.log_latent_muvar = np.array([0.] * self.global_latent_len)
        self.log_latent_logsig = np.array([0.] * self.global_latent_len)
        self.log_num = 0

    def network_update(self):
        self.log_encoder_loss_sim = np.array([0.] * self.global_latent_len)
        self.log_encoder_loss_dis = np.array([0.] * self.global_latent_len)
        self.log_encoder_loss_kl = 0.
        self.log_encoder_loss_l2 = 0.
        self.log_route_loss_rec = np.array([0.] * self.nextstate_len)
        self.log_route_loss_l2 = 0.
        self.log_action_loss_rec = 0.
        self.log_action_loss_l2 = 0.
        self.log_latent_mumean = np.array([0.] * self.global_latent_len)
        self.log_latent_muvar = np.array([0.] * self.global_latent_len)
        self.log_latent_logsig = np.array([0.] * self.global_latent_len)
        self.log_num = 0
            
    def optimize(self, iteration_num, input_state, input_nextstate, input_route, input_action):
        input_list = {self.layer_iteration_num : iteration_num, self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate,
                      self.layer_input_action : input_action, self.layer_input_route : input_route, self.layer_dropout : 0.1 }
        sess = tf.get_default_session()
        _, l1, l1_2, l2, l3, l4, l5, l6, l7, l8, l9, l10 = sess.run([self.train_action, self.global_latent_similarity, self.global_latent_difference,
                                             self.encoder_kl_loss, self.encoder_l2_loss,
                                              self.route_decoder_reconstruction_loss, self.route_decoder_l2_loss,
                                              self.action_decoder_reconstruction_loss, self.action_decoder_l2_loss,
                                              self.latent_mumean, self.latent_muvar, self.latent_logsig],input_list)

        self.log_encoder_loss_sim += l1
        self.log_encoder_loss_dis += l1_2
        self.log_encoder_loss_kl += l2
        self.log_encoder_loss_l2 += l3
        self.log_route_loss_rec += l4
        self.log_route_loss_l2 += l5
        self.log_action_loss_rec += l6
        self.log_action_loss_l2 += l7
        self.log_latent_mumean += l8
        self.log_latent_muvar += l9
        self.log_latent_logsig += l10
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.global_encoder.mu, self.global_encoder.sig], input_list)
        return l1, l2


    def get_global_decoded(self, input_state, input_route, input_global_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_route : input_route, self.layer_input_global_latent : input_global_latent, self.layer_dropout : 0.0}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.latent_route_decoder_output, self.latent_action], input_list)
        return l1, l2

       
    def log_caption(self):
        return "\t" + self.name + "_GlobalSimLoss\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) + self.name + "_GlobalDisLoss\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_GlobalKLLoss\t" + self.name + "_GlobalL2Loss\t" \
            + self.name + "_RouteRegLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) + self.name + "_RouteL2Loss\t" \
            + self.name + "_ActionRegLoss\t" + self.name + "_ActionL2Loss\t" \
            + self.name + "_LatentMuMean\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_LatentMuVar\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_LatentLogSig\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) 

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_encoder_loss_sim[i] / log_num) for i in range(self.global_latent_len)])  + "\t"\
            + "\t".join([str(self.log_encoder_loss_dis[i] / log_num) for i in range(self.global_latent_len)])  + "\t"\
            + str(self.log_encoder_loss_kl / log_num) + "\t" + str(self.log_encoder_loss_l2 / log_num) + "\t"\
            + "\t".join([str(self.log_route_loss_rec[i] / log_num) for i in range(self.nextstate_len)])  + "\t"\
            + str(self.log_route_loss_l2 / log_num)+ "\t" \
            + str(self.log_action_loss_rec / log_num)+ "\t" \
            + str(self.log_action_loss_l2 / log_num)+ "\t" \
            + "\t".join([str(self.log_latent_mumean[i] / log_num) for i in range(self.global_latent_len)])+ "\t" \
            + "\t".join([str(self.log_latent_muvar[i] / log_num) for i in range(self.global_latent_len)])+ "\t" \
            + "\t".join([str(self.log_latent_logsig[i] / log_num) for i in range(self.global_latent_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tGlobalSimLoss        : " + " ".join([str(self.log_encoder_loss_sim[i] / log_num)[:8] for i in range(self.global_latent_len)]) \
            + "\n\tGlobalDisLoss        : " + " ".join([str(self.log_encoder_loss_dis[i] / log_num)[:8] for i in range(self.global_latent_len)]) \
            + "\n\tGlobalRegLoss        : " + str(self.log_encoder_loss_kl / log_num) \
            + "\n\tGlobalL2Loss         : " + str(self.log_encoder_loss_l2 / log_num) \
            + "\n\tRouteReconLoss       : " + " ".join([str(self.log_route_loss_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tRouteL2Loss          : " + str(self.log_route_loss_l2 / log_num) \
            + "\n\tActionReconLoss      : " + str(self.log_action_loss_rec / log_num) \
            + "\n\tActionL2Loss         : " + str(self.log_action_loss_l2 / log_num) \
            + "\n\tLatentMuMean         : " + " ".join([str(self.log_latent_mumean[i] / log_num)[:8] for i in range(self.global_latent_len)])\
            + "\n\tLatentMuvar          : " + " ".join([str(self.log_latent_muvar[i] / log_num)[:8] for i in range(self.global_latent_len)])\
            + "\n\tLatentLogSig         : " + " ".join([str(self.log_latent_logsig[i] / log_num)[:8] for i in range(self.global_latent_len)]))
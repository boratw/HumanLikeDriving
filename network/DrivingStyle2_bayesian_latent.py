import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, regularizer_weight= 0.001,
                 lr = 0.001, not_action_lr = 0.05, latent_len=4, decoder_shuffle_num=7):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.nextstate_len = nextstate_len
        self.route_len = route_len
        self.action_len = action_len
        self.latent_len = latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_route = tf.placeholder(tf.float32, [None, action_len, route_len])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)
            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])
            self.action_normalizer = tf.get_variable("action_normalizer", dtype=tf.float32, 
                initializer=tf.ones([action_len]) / action_len, trainable=True)
            

            self.all_route_input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            self.enc_h1 = Bayesian_FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="enc_h1")
            self.enc_h2 = Bayesian_FC(self.enc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="enc_h2")
            self.l_latent = Bayesian_FC(self.enc_h2.layer_output, 128, latent_len, input_dropout = None, 
                                  output_nonln = None, name="l_latent")
            self.latent = self.l_latent.layer_output
            batched_global_latent = tf.tile(self.latent, [decoder_shuffle_num+1, 1])
            shuffled_global_latent = tf.tile(self.latent, [decoder_shuffle_num, 1])
            tf.random.shuffle(shuffled_global_latent)
            self.latent_decoder_input = tf.concat([self.latent, shuffled_global_latent], axis=0)
            self.state_decoder_input = tf.tile(self.all_route_input, [decoder_shuffle_num+1, 1])
            self.decoder_input =  tf.concat([self.latent_decoder_input, self.state_decoder_input], axis=1)

            self.dec_h1 = Bayesian_FC(self.decoder_input, state_len + action_len * route_len + latent_len, 256, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="dec_h1")
            self.l_action = Bayesian_FC(self.dec_h1.layer_output, 256, action_len, input_dropout = None, 
                                  output_nonln = None, name="l_action")
            self.output_action = tf.nn.softmax(self.l_action.layer_output, axis=1)
            self.dec_h2 = [Bayesian_FC(self.dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="dec_h2_" + str(i)) for i in range(action_len)]
            self.l_route = [Bayesian_FC(self.dec_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                  output_nonln = None, name="l_route" + str(i)) for i in range(action_len)]
            self.mixed_output_route = tf.stack([self.l_route[i].layer_output for i in range(action_len)], axis=1)

            batched_input_nextstate =  tf.tile(self.layer_input_nextstate, [decoder_shuffle_num + 1, 1])
            self.mixed_route_error = tf.reduce_mean((self.mixed_output_route - tf.reshape(batched_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
            self.mixed_minimum_loss_action = tf.math.argmin(self.mixed_route_error, axis=1)
            self.mixed_minimum_route_error = (tf.gather(self.mixed_output_route, self.mixed_minimum_loss_action, axis=1, batch_dims=1) - batched_input_nextstate) ** 2
            
            self.route_error_vec = tf.split(self.mixed_route_error, decoder_shuffle_num+1, axis=0)
            self.minimum_route_error_vec = tf.split(self.mixed_minimum_route_error, decoder_shuffle_num+1, axis=0)
            self.minimum_loss_action_vec = tf.split(self.mixed_minimum_loss_action, decoder_shuffle_num+1, axis=0)
            self.output_action_vec = tf.split(self.output_action, decoder_shuffle_num+1, axis=0)

            self.maximum_output_action = tf.reduce_mean(tf.reduce_max(self.output_action_vec[0], axis=1))
            self.route_error = self.route_error_vec[0]
            self.minimum_loss_action = self.minimum_loss_action_vec[0]
            self.minimum_route_error = tf.reduce_mean(self.minimum_route_error_vec[0], axis=0)
            self.minimum_not_route_error = tf.reduce_mean(self.minimum_route_error_vec[1:], axis=[0, 1])

            self.mixed_latent_error = 1. / (tf.reduce_mean(self.mixed_minimum_route_error, axis=1) + 0.001)
            self.latent_error = tf.reduce_mean(-(batched_global_latent / (tf.math.sqrt(tf.reduce_sum(batched_global_latent ** 2, axis=1, keepdims=True)) + 1e-5)) * 
                                               tf.stop_gradient(self.latent_decoder_input / (tf.math.sqrt(tf.reduce_sum(self.latent_decoder_input ** 2, axis=1, keepdims=True)) + 1e-5)),
                                               axis=1) * tf.stop_gradient(self.mixed_latent_error)

            self.average_latent_mean = tf.reduce_mean(self.l_latent.layer_mean, axis=0)
            self.average_latent_var = tf.reduce_mean(self.l_latent.layer_var, axis=0)

            self.latent_reg_loss = self.enc_h1.regularization_loss + self.enc_h2.regularization_loss + self.l_latent.regularization_loss
            self.route_reg_loss = self.dec_h1.regularization_loss
            for i in range(action_len):
                self.route_reg_loss += self.dec_h2[i].regularization_loss
                self.route_reg_loss += self.l_route[i].regularization_loss
            self.route_reg_loss /= (action_len * 2. + 1)
            self.action_reg_loss = (self.dec_h1.regularization_loss + self.l_action.regularization_loss) / 2.

            self.optimizer = tf.train.AdamOptimizer(lr)

            self.train_latent = self.optimizer.minimize(loss = self.latent_error + self.latent_reg_loss * regularizer_weight,
                                                       var_list=[*self.enc_h1.trainable_params, *self.enc_h2.trainable_params, *self.l_latent.trainable_params] )  
            
            train_route_vars = [*self.dec_h1.trainable_params]
            for i in range(action_len):
                train_route_vars.extend(self.dec_h2[i].trainable_params)
                train_route_vars.extend(self.l_route[i].trainable_params)
            self.train_route = self.optimizer.minimize(loss = tf.reduce_mean(self.minimum_route_error) + tf.reduce_mean(self.route_error) * not_action_lr
                                                       + self.route_reg_loss * regularizer_weight,
                                                       var_list=train_route_vars )
            action_possibility = self.action_normalizer / self.route_error
            action_possibility_softmax = action_possibility / tf.reduce_sum(action_possibility, axis=1, keepdims=True)
            self.action_error = - tf.stop_gradient(action_possibility_softmax) * self.output_action_vec[0]
            self.train_action = self.optimizer.minimize(loss = self.action_error + self.action_reg_loss * regularizer_weight,
                                                       var_list=[*self.dec_h1.trainable_params, *self.l_action.trainable_params] )  
               

            normalizer_loss = tf.gather(( tf.stop_gradient(self.route_error) - self.action_normalizer) ** 2, self.minimum_loss_action, axis=1, batch_dims=1)
            self.train_normalizer = self.optimizer.minimize(loss = normalizer_loss, var_list=[self.action_normalizer] )

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_var = np.array([0.] * self.latent_len)
        self.log_rec = np.array([0.] * self.nextstate_len)
        self.log_not_rec = np.array([0.] * self.nextstate_len)
        self.log_action = 0.
        self.log_l_reg = 0.
        self.log_a_reg = 0.
        self.log_r_reg = 0.
        self.log_a_norm = np.array([0.] * self.action_len)
        self.log_num = 0

    def network_update(self):
        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_var = np.array([0.] * self.latent_len)
        self.log_rec = np.array([0.] * self.nextstate_len)
        self.log_not_rec = np.array([0.] * self.nextstate_len)
        self.log_action = 0.
        self.log_l_reg = 0.
        self.log_a_reg = 0.
        self.log_r_reg = 0.
        self.log_a_norm = np.array([0.] * self.action_len)
        self.log_num = 0

            
    def optimize(self, input_state, input_nextstate, input_route):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, _, _, _, l1, l2, l3, l4, l5, l6, l7, l8, l9  = sess.run([self.train_latent, self.train_route, self.train_action, self.train_normalizer, 
                                       self.minimum_route_error, self.maximum_output_action, self.action_reg_loss, self.route_reg_loss, 
                                       self.action_normalizer, self.average_latent_mean, self.average_latent_var, self.minimum_not_route_error,
                                       self.latent_reg_loss  ],input_list)
        
        self.log_latent_mean += l6
        self.log_latent_var += l7
        self.log_rec += l1
        self.log_not_rec += l8
        self.log_action += l2
        self.log_l_reg += l9
        self.log_a_reg += l3
        self.log_r_reg += l4
        self.log_a_norm += l5
        self.log_num += 1
       
    def log_caption(self):
        return "\t" + self.name + "_LatentMean\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentVar\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_ReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_NotLatentReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_MaximumAction\t"  \
            + self.name + "_LatentRegLoss\t"  \
            + self.name + "_ActionRegLoss\t"  \
            + self.name + "_RouteRegLoss\t"  \
            + self.name + "_AverageAction\t" + "\t".join([ "" for _ in range(self.action_len)])

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_latent_mean[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_var[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_not_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_action / log_num) + "\t"\
            + str(self.log_l_reg / log_num) + "\t"\
            + str(self.log_a_reg / log_num) + "\t"\
            + str(self.log_r_reg / log_num) + "\t"\
            + "\t".join([str(self.log_a_norm[i] / log_num) for i in range(self.action_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tLatentMean          : " + " ".join([str(self.log_latent_mean[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLatentVar           : " + " ".join([str(self.log_latent_var[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tReconLoss           : " + " ".join([str(self.log_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tNotLatentReconLoss  : " + " ".join([str(self.log_not_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tMaximumAction       : " + str(self.log_action / log_num) \
            + "\n\tLatentRegLoss       : " + str(self.log_l_reg / log_num) \
            + "\n\tActionRegLoss       : " + str(self.log_a_reg / log_num) \
            + "\n\tRouteRegLoss        : " + str(self.log_r_reg / log_num) \
            + "\n\tAverageAction       : " + " ".join([str(self.log_a_norm[i] / log_num) for i in range(self.action_len)]) )
        
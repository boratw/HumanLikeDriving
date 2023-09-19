import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC, FC, Variational_FC



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, latent_len=4, regularizer_weight= 0.001,
                 discrete_lr = 0.001, latent_lr = 0.001,  istraining=True,
                 num_of_agents=4):

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
            self.layer_input_latent = tf.placeholder(tf.float32, [None, latent_len])
            self.layer_input_action = tf.placeholder(tf.int32, [None])
            self.layer_input_index = tf.placeholder(tf.int32, [num_of_agents])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)

            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])
            self.all_route_input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            self.action_label = tf.one_hot(self.layer_input_action, action_len)
            with tf.variable_scope("DiscreteLearner", reuse=reuse):

                self.ds_action_normalizer = tf.get_variable("action_normalizer", dtype=tf.float32, 
                    initializer=tf.ones([action_len]) / action_len, trainable=True)
            
                self.ds_h1 = FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="ds_h1")
                self.ds_h2 = [FC(self.ds_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="ds_h2_" + str(i)) for i in range(action_len)]
                self.ds_route = [FC(self.ds_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                    output_nonln = None, name="ds_route_" + str(i)) for i in range(action_len)]
                self.ds_output_route = tf.stack([self.ds_route[i].layer_output for i in range(action_len)], axis=1)

                self.ds_route_error = tf.reduce_mean((self.ds_output_route - tf.reshape(self.layer_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
                self.ds_min_route_diff = tf.gather(self.ds_output_route, self.layer_input_action, axis=1, batch_dims=1) - self.layer_input_nextstate
                self.ds_min_route_error = tf.reduce_mean(tf.abs(self.ds_min_route_diff), axis=0)
                self.ds_min_route_loss = tf.reduce_mean(self.ds_min_route_diff ** 2)


                self.ds_optimizer = tf.train.AdamOptimizer(discrete_lr)
                train_route_vars = [*self.ds_h1.trainable_params]
                for i in range(action_len):
                    train_route_vars.extend(self.ds_h2[i].trainable_params)
                    train_route_vars.extend(self.ds_route[i].trainable_params)
                self.ds_train_route = self.ds_optimizer.minimize(loss = self.ds_min_route_loss, var_list=train_route_vars )

                self.ds_params = [*self.ds_h1.trainable_params]
                for i in range(action_len):
                    self.ds_params.extend(self.ds_h2[i].trainable_params)
                    self.ds_params.extend(self.ds_route[i].trainable_params)

                self.ds_action_route_error = tf.reduce_mean(self.ds_route_error, axis=0)


            with tf.variable_scope("DiscreteLearner_copied", reuse=reuse):
                self.dsc_h1 = FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="dsc_h1")
                self.dsc_h2 = [FC(self.dsc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="dsc_h2_" + str(i)) for i in range(action_len)]
                self.dsc_route = [FC(self.dsc_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                    output_nonln = None, name="dsc_route_" + str(i)) for i in range(action_len)]
                self.dsc_output_route = tf.stack([self.dsc_route[i].layer_output for i in range(action_len)], axis=1)

                self.dsc_route_error = tf.reduce_mean((self.dsc_output_route - tf.reshape(self.layer_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
                self.dsc_min_route_diff = tf.gather(self.dsc_output_route, self.layer_input_action, axis=1, batch_dims=1) - self.layer_input_nextstate


                self.dsc_params = [*self.dsc_h1.trainable_params]
                for i in range(action_len):
                    self.dsc_params.extend(self.dsc_h2[i].trainable_params)
                    self.dsc_params.extend(self.dsc_route[i].trainable_params)
                self.dsc_copy_ds = [ tf.assign(target, source) for target, source in zip(self.dsc_params, self.ds_params)]
                self.dsc_add_ds = [ tf.assign(target, target * 0.96 + source * 0.04) for target, source in zip(self.dsc_params, self.ds_params)]

            if istraining:
                with tf.variable_scope("LatentLearner", reuse=reuse):
                    self.ls_encoder_input = tf.concat([self.all_route_input, self.action_label, tf.stop_gradient(self.dsc_min_route_diff)], axis=1)

                    self.ls_enc_h1 = Bayesian_FC(self.ls_encoder_input, state_len + action_len * route_len + action_len + nextstate_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_enc_h1")
                    self.ls_enc_h2 = Bayesian_FC(self.ls_enc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_enc_h2")
                    self.ls_latent = FC(self.ls_enc_h2.layer_output, 128, latent_len, input_dropout = None, 
                                        output_nonln = None, name="ls_latent")
                    
                    self.ls_output_latent = self.ls_latent.layer_output
                    output_latents = tf.split(self.ls_output_latent, self.layer_input_index, axis=0)

                    self.ls_decoder_input_latents = []
                    for output_latent in output_latents:
                        output_latent_count = tf.shape(output_latent)[0]
                        output_latent_sum = tf.reduce_sum(output_latent, axis=0, keepdims=True)
                        output_latent_norm = output_latent_sum / tf.stop_gradient(tf.sqrt(tf.reduce_sum(output_latent_sum ** 2, axis=1, keepdims=True) + 1e-6))
                        self.ls_decoder_input_latents.append(tf.tile(tf.reshape(output_latent_norm, [1, latent_len]), [output_latent_count, 1]))
                    self.ls_decoder_input_latent = tf.concat(self.ls_decoder_input_latents, axis=0)

                    self.ls_latent_variance = tf.math.reduce_variance(self.ls_output_latent, axis=0)

                    self.ls_decoder_input = tf.concat([self.all_route_input, self.ls_decoder_input_latent], axis=1)
                    self.ls_dec_h1 = Bayesian_FC(self.ls_decoder_input, state_len + action_len * route_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1")
                    self.ls_action = FC(self.ls_dec_h1.layer_output, 256, action_len, input_dropout = None, 
                                        output_nonln = None, name="ls_action")
                    self.ls_dec_h2 = [Bayesian_FC(self.ls_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2_" + str(i)) for i in range(action_len)]
                    self.ls_diff = [FC(self.ls_dec_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff_" + str(i)) for i in range(action_len)]
                    
                    self.ls_output_action = tf.nn.softmax(self.ls_action.layer_output, axis=1)
                    self.ls_output_diff = tf.stack([self.ls_diff[i].layer_output for i in range(action_len)], axis=1)
                    self.ls_output_min_diff = tf.gather(self.ls_output_diff, self.layer_input_action, axis=1, batch_dims=1)
                    self.ls_route_error = tf.reduce_mean(tf.abs(self.ls_output_min_diff - self.dsc_min_route_diff), axis=0)
                    self.ls_route_loss = tf.reduce_mean((self.ls_output_min_diff - self.dsc_min_route_diff) ** 2)
                    self.ls_action_loss = tf.reduce_mean(-tf.log(self.ls_output_action + 1e-6) * self.action_label)

                    self.ls_route_optimizer = tf.train.AdamOptimizer(latent_lr)
                    self.ls_action_optimizer = tf.train.AdamOptimizer(latent_lr)
                    self.ls_enc_reg_loss = (self.ls_enc_h1.regularization_loss + self.ls_enc_h2.regularization_loss +
                                            self.ls_latent.regularization_loss) / 3
                    self.ls_dec_action_reg_loss = (self.ls_dec_h1.regularization_loss + self.ls_action.regularization_loss) / 2
                    self.ls_dec_route_reg_loss = self.ls_dec_h1.regularization_loss
                    for i in range(action_len):
                        self.ls_dec_route_reg_loss += self.ls_dec_h2[i].regularization_loss
                        self.ls_dec_route_reg_loss += self.ls_diff[i].regularization_loss
                    self.ls_dec_route_reg_loss = self.ls_dec_route_reg_loss / (action_len * 2 + 2)
                    self.ls_reg_loss = self.ls_enc_reg_loss + self.ls_dec_route_reg_loss + self.ls_dec_action_reg_loss
                    
                    self.ls_average_action_mean = tf.reduce_mean(self.ls_output_action, axis=0)
                    self.ls_average_latent_mean = tf.reduce_mean(self.ls_output_latent ** 2, axis=0)
                    self.ls_latent_reg_loss = tf.reduce_mean(self.ls_average_latent_mean)

                    self.ls_train_route = self.ls_route_optimizer.minimize(loss = 
                                                            self.ls_route_loss +
                                                            self.ls_enc_reg_loss * regularizer_weight +
                                                            self.ls_dec_route_reg_loss * regularizer_weight +
                                                            self.ls_latent_reg_loss * regularizer_weight,
                                                            var_list = tf.trainable_variables(scope=tf.get_variable_scope().name) )   
                    self.ls_train_action = self.ls_action_optimizer.minimize(loss = 
                                                            self.ls_action_loss +
                                                            self.ls_enc_reg_loss * regularizer_weight +
                                                            self.ls_dec_action_reg_loss * regularizer_weight +
                                                            self.ls_latent_reg_loss * regularizer_weight,
                                                            var_list = tf.trainable_variables(scope=tf.get_variable_scope().name) )   
            else:
                with tf.variable_scope("LatentLearner", reuse=reuse):
                    self.ls_encoder_input = tf.concat([self.all_route_input, self.action_label, tf.stop_gradient(self.dsc_min_route_diff)], axis=1)

                    self.ls_enc_h1 = Bayesian_FC(self.ls_encoder_input, state_len + action_len * route_len + action_len + nextstate_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_enc_h1")
                    self.ls_enc_h2 = Bayesian_FC(self.ls_enc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_enc_h2")
                    self.ls_latent = FC(self.ls_enc_h2.layer_output, 128, latent_len, input_dropout = None, 
                                        output_nonln = None, name="ls_latent")
                    self.ls_output_latent = self.ls_latent.layer_output
                    
                    self.ls_decoder_input = tf.concat([self.all_route_input, self.ls_output_latent], axis=1)
                    self.ls_dec_h1 = Bayesian_FC(self.ls_decoder_input, state_len + action_len * route_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1")
                    self.ls_action = FC(self.ls_dec_h1.layer_output, 256, 2, input_dropout = None, 
                                        output_nonln = None, name="ls_action")
                    self.ls_dec_h2 = [Bayesian_FC(self.ls_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2_" + str(i)) for i in range(action_len)]
                    self.ls_diff = [FC(self.ls_dec_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff_" + str(i)) for i in range(action_len)]
                    
                    self.ls_infer_decoder_input = tf.concat([self.all_route_input, self.layer_input_latent], axis=1)
                    self.ls_infer_dec_h1 = Bayesian_FC(self.ls_infer_decoder_input, state_len + action_len * route_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1", reuse=True)
                    self.ls_infer_action = FC(self.ls_infer_dec_h1.layer_output, 256, 2, input_dropout = None, 
                                        output_nonln = None, name="ls_action", reuse=True)
                    self.ls_infer_dec_h2 = [Bayesian_FC(self.ls_infer_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2_" + str(i), reuse=True ) for i in range(action_len)]
                    self.ls_infer_diff = [FC(self.ls_infer_dec_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff_" + str(i), reuse=True) for i in range(action_len)]
                    

                    self.ls_infer_output_diff = tf.stack([self.ls_infer_diff[i].mu for i in range(action_len)], axis=1)

                    self.output_route_mean = self.dsc_output_route# - self.ls_infer_output_diff
                    self.output_route_var = tf.stack([self.ls_infer_diff[i].var for i in range(action_len)], axis=1)

                    self.output_action = tf.nn.softmax(self.ls_infer_action.layer_output, axis=1)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_rec = np.array([0.] * self.nextstate_len)
        self.log_latent_action_loss = 0.
        self.log_latent_action_mean = np.array([0., 0.])
        self.log_latent_merge_loss = 0.
        self.log_latent_diverse = 0.
        self.log_latent_reg = 0.
        self.log_disc_rec = np.array([0.] * self.nextstate_len)
        self.log_disc_action_mean = np.array([0.] * self.action_len)
        self.log_num = 0

        sess = tf.get_default_session()
        sess.run(self.dsc_copy_ds)


    def network_update(self):
        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_rec = np.array([0.] * self.nextstate_len)
        self.log_latent_action_loss = 0.
        self.log_latent_action_mean = np.array([0., 0.])
        self.log_latent_reg = 0.
        self.log_latent_diverse = np.array([0.] * self.latent_len)
        self.log_disc_rec = np.array([0.] * self.nextstate_len)
        self.log_disc_action_mean = np.array([0.] * self.action_len)
        self.log_num = 0


                     
    def optimize(self, input_state, input_nextstate, input_route, input_action, input_index):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_action : input_action, self.layer_input_index : input_index, self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, l6, l8 =\
                sess.run([self.ds_train_route,
                          self.ds_min_route_error, self.ds_action_route_error],input_list)
        
        _, _, l1, l10, l4, l5, l7, l11 =\
                sess.run([self.ls_train_route, self.ls_train_action,
                          self.ls_average_latent_mean, self.ls_route_error,
                          self.ls_latent_variance,
                          self.ls_reg_loss, self.ls_action_loss, self.ls_average_action_mean],input_list)
        
        
        self.log_latent_mean += l1
        self.log_latent_rec += l10
        self.log_latent_action_loss += l7
        self.log_latent_action_mean += l11
        self.log_latent_diverse += l4
        self.log_latent_reg += l5
        self.log_disc_rec += l6
        self.log_disc_action_mean += l8
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate, input_route, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_action : input_action, self.layer_input_dropout : 0.}
        sess = tf.get_default_session()
        l1 = sess.run(self.ls_latent.layer_output, input_list)
        return l1
    
    def get_output(self, input_state, input_route, input_latent, input_dropout=0.0):
        input_list = {self.layer_input_state : input_state, self.layer_input_route : input_route, self.layer_input_latent : input_latent,
                      self.layer_input_dropout : input_dropout}
        sess = tf.get_default_session()
        l1, l2, l3 = sess.run([self.output_route_mean, self.output_route_var, self.output_action], input_list)
        return l1, l2, l3

    def optimize_update(self):
        sess = tf.get_default_session()
        sess.run(self.dsc_add_ds)

    def log_caption(self):
        return "\t" + self.name + "_LatentMean\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_LatentActionLoss\t"  \
            + self.name + "_LatentActionMean\t\t"  \
            + self.name + "_LatentDiverse\t"  + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentRegLoss\t"  \
            + self.name + "_DiscReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_AverageRouteError\t" + "\t".join([ "" for _ in range(self.action_len)])  

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_latent_mean[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_latent_action_loss / log_num) + "\t"\
            + "\t".join([str(self.log_latent_action_mean[i] / log_num) for i in range(2)]) + "\t"\
            + "\t".join([str(self.log_latent_diverse[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + str(self.log_latent_reg / log_num) + "\t"\
            + "\t".join([str(self.log_disc_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_disc_action_mean[i] / log_num) for i in range(self.action_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tLatentMean          : " + " ".join([str(self.log_latent_mean[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLatentReconLoss     : " + " ".join([str(self.log_latent_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tLatentActionLoss    : " + str(self.log_latent_action_loss / log_num) \
            + "\n\tLatentActionMean    : " + " ".join([str(self.log_latent_action_mean[i] / log_num)[:8] for i in range(2)]) \
            + "\n\tLatentDiverse       : " + " ".join([str(self.log_latent_diverse[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLatentRegLoss       : " + str(self.log_latent_reg / log_num) \
            + "\n\tDiscReconLoss       : " + " ".join([str(self.log_disc_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tAverageRouteError   : " + " ".join([str(self.log_disc_action_mean[i] / log_num) for i in range(self.action_len)]) )
        
import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC, FC



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, latent_len=4, regularizer_weight= 0.0001,
                 discrete_lr = 0.001, discrete_action_norm_lr = 0.0001, latent_lr = 0.001, shuffled_lr=0.25, istraining=True,
                 divloss_weight = 0.001, mu_loss_weight=0.01):

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
            self.layer_input_dropout = tf.placeholder(tf.float32, None)

            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])
            self.all_route_input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            with tf.variable_scope("DiscreteLearner", reuse=reuse):

                self.ds_action_normalizer = tf.get_variable("action_normalizer", dtype=tf.float32, 
                    initializer=tf.ones([action_len]) / action_len, trainable=True)
            
                self.ds_h1 = FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="ds_h1")
                self.ds_action = FC(self.ds_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="ds_action")
                self.ds_output_action = tf.nn.softmax(self.ds_action.layer_output, axis=1)
                self.ds_h2 = [FC(self.ds_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="ds_h2_" + str(i)) for i in range(action_len)]
                self.ds_route = [FC(self.ds_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                    output_nonln = None, name="ds_route_" + str(i)) for i in range(action_len)]
                self.ds_output_route = tf.stack([self.ds_route[i].layer_output for i in range(action_len)], axis=1)

                self.ds_route_error = tf.reduce_mean((self.ds_output_route - tf.reshape(self.layer_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
                self.ds_min_route_diff = tf.gather(self.ds_output_route, self.layer_input_action, axis=1, batch_dims=1) - self.layer_input_nextstate
                self.ds_min_route_error = tf.reduce_mean(tf.abs(self.ds_min_route_diff), axis=0)


                self.ds_optimizer = tf.train.AdamOptimizer(discrete_lr)
                train_route_vars = [*self.ds_h1.trainable_params]
                for i in range(action_len):
                    train_route_vars.extend(self.ds_h2[i].trainable_params)
                    train_route_vars.extend(self.ds_route[i].trainable_params)
                self.ds_train_route = self.ds_optimizer.minimize(loss = tf.reduce_mean(self.ds_min_route_error), var_list=train_route_vars )

                action_label = tf.one_hot(self.layer_input_action, action_len)
                self.action_error = -action_label * self.ds_output_action
                self.ds_train_action = self.ds_optimizer.minimize(loss = self.action_error,
                                                        var_list=[*self.ds_h1.trainable_params, *self.ds_action.trainable_params] )     

                self.ds_params = [*self.ds_h1.trainable_params, *self.ds_action.trainable_params]
                for i in range(action_len):
                    self.ds_params.extend(self.ds_h2[i].trainable_params)
                    self.ds_params.extend(self.ds_route[i].trainable_params)

                self.ds_average_action = tf.reduce_mean(self.ds_output_action, axis=0)
                self.ds_action_route_error = tf.reduce_mean(self.ds_route_error, axis=0)


            with tf.variable_scope("DiscreteLearner_copied", reuse=reuse):
                self.dsc_h1 = FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="dsc_h1")
                self.dsc_action = FC(self.dsc_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="dsc_action")
                self.dsc_output_action = tf.nn.softmax(self.dsc_action.layer_output, axis=1)
                self.dsc_h2 = [FC(self.dsc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="dsc_h2_" + str(i)) for i in range(action_len)]
                self.dsc_route = [FC(self.dsc_h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                    output_nonln = None, name="dsc_route_" + str(i)) for i in range(action_len)]
                self.dsc_output_route = tf.stack([self.dsc_route[i].layer_output for i in range(action_len)], axis=1)

                self.dsc_route_error = tf.reduce_mean((self.dsc_output_route - tf.reshape(self.layer_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
                self.dsc_min_route_diff = tf.gather(self.dsc_output_route, self.layer_input_action, axis=1, batch_dims=1) - self.layer_input_nextstate


                self.dsc_params = [*self.dsc_h1.trainable_params, *self.dsc_action.trainable_params]
                for i in range(action_len):
                    self.dsc_params.extend(self.dsc_h2[i].trainable_params)
                    self.dsc_params.extend(self.dsc_route[i].trainable_params)
                self.dsc_copy_ds = [ tf.assign(target, source) for target, source in zip(self.dsc_params, self.ds_params)]
                self.dsc_add_ds = [ tf.assign(target, target * 0.96 + source * 0.04) for target, source in zip(self.dsc_params, self.ds_params)]

            if istraining:
                with tf.variable_scope("LatentLearner", reuse=reuse):
                    self.ls_encoder_input = tf.concat([self.all_route_input, action_label, tf.stop_gradient(self.dsc_min_route_diff)], axis=1)

                    self.ls_enc_h1 = Bayesian_FC(self.ls_encoder_input, state_len + action_len * route_len + action_len + nextstate_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_enc_h1")
                    self.ls_enc_h2 = Bayesian_FC(self.ls_enc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_enc_h2")
                    self.ls_latent = FC(self.ls_enc_h2.layer_output, 128, latent_len * 2, input_dropout = None, 
                                        output_nonln = None, name="ls_latent")
                    self.ls_latent_mu, ls_latent_logsig = tf.split(self.ls_latent.layer_output, 2, axis=1)
                    self.ls_latent_logsig = tf.clip_by_value(ls_latent_logsig, -10, 2) - 2.
                    self.ls_latent_sig = tf.exp(self.ls_latent_logsig)
                    noise = tf.random.normal(tf.shape(self.ls_latent_mu))

                    self.ls_output_latent = self.ls_latent_mu + self.ls_latent_sig * noise
                    self.ls_latent_var = self.ls_latent_sig ** 2

                    latent_var = tf.reduce_mean(self.ls_latent_var, axis=1)
                    sorted_indicies = tf.argsort(latent_var, axis=0, direction='ASCENDING')
                    indicies_num = tf.shape(sorted_indicies)[0]
                    indicies1 = tf.random.shuffle(tf.tile(sorted_indicies[:indicies_num // 2], [2]))
                    shuffled_latent1 = tf.gather(self.ls_output_latent, indicies1, axis=0, batch_dims=0)
                    indicies2 = tf.random.shuffle(tf.tile(sorted_indicies[:indicies_num // 4], [4]))
                    shuffled_latent2 = tf.gather(self.ls_output_latent, indicies2, axis=0, batch_dims=0)
                    indicies3 = tf.random.shuffle(tf.tile(sorted_indicies[:indicies_num // 8], [8]))
                    shuffled_latent3 = tf.gather(self.ls_output_latent, indicies3, axis=0, batch_dims=0)

                    self.ls_batched_output_latent = tf.concat([self.ls_output_latent, shuffled_latent1, shuffled_latent2, shuffled_latent3], axis=0)
                    self.ls_batched_route_input = tf.tile(self.all_route_input, [4, 1])
                    action_label_batched = tf.tile(action_label, [4, 1])

                    self.ls_decoder_input = tf.concat([self.ls_batched_route_input, action_label_batched, self.ls_batched_output_latent], axis=1)
                    self.ls_dec_h1 = Bayesian_FC(self.ls_decoder_input, state_len + action_len * route_len + action_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1")
                    self.ls_dec_h2 = Bayesian_FC(self.ls_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2")
                    self.ls_diff = Bayesian_FC(self.ls_dec_h2.layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff")
                
                    self.ls_batched_route_diff = tf.stop_gradient(tf.tile(self.dsc_min_route_diff, [4, 1]))

                    self.ls_route_error = tf.abs(self.ls_diff.layer_output - self.ls_batched_route_diff)
                    ls_route_errors = tf.split(self.ls_route_error, 4)
                    self.ls_route_loss1 = tf.reduce_mean(ls_route_errors[0], axis=0)
                    self.ls_route_loss2 = tf.reduce_mean(ls_route_errors[1], axis=0)
                    self.ls_route_loss3 = tf.reduce_mean(ls_route_errors[2], axis=0)
                    self.ls_route_loss4 = tf.reduce_mean(ls_route_errors[3], axis=0)

                    self.ls_optimizer = tf.train.AdamOptimizer(latent_lr)
                    self.ls_enc_reg_loss = (self.ls_enc_h1.regularization_loss + self.ls_enc_h2.regularization_loss) / 2
                    self.ls_dec_reg_loss = (self.ls_dec_h1.regularization_loss + self.ls_dec_h2.regularization_loss) / 2
                    self.ls_reg_loss = self.ls_enc_reg_loss + self.ls_dec_reg_loss
                    
                    self.ls_div_loss = tf.reduce_mean(self.ls_latent_var + self.ls_latent_mu ** 2 - (self.ls_latent_logsig - 2.))
                    self.ls_mu_loss = tf.reduce_mean(self.ls_latent_mu ** 2)
                    self.ls_train_latent_enc = self.ls_optimizer.minimize(loss = 
                                                            tf.reduce_mean(ls_route_errors[0]) +
                                                            tf.reduce_mean(ls_route_errors[1]) * shuffled_lr +
                                                            tf.reduce_mean(ls_route_errors[2]) * shuffled_lr +
                                                            tf.reduce_mean(ls_route_errors[3]) * shuffled_lr +
                                                            self.ls_enc_reg_loss * regularizer_weight +
                                                            self.ls_div_loss * divloss_weight +
                                                            self.ls_mu_loss * mu_loss_weight,
                                                            var_list=[*self.ls_enc_h1.trainable_params, *self.ls_enc_h2.trainable_params,
                                                                      *self.ls_latent.trainable_params] )     
                    self.ls_train_latent_dec = self.ls_optimizer.minimize(loss = 
                                                            tf.reduce_mean(ls_route_errors[0]) +
                                                            self.ls_dec_reg_loss * regularizer_weight,
                                                            var_list=[*self.ls_dec_h1.trainable_params,
                                                                      *self.ls_dec_h2.trainable_params, *self.ls_diff.trainable_params] )    
                    self.ls_average_latent_mean = tf.reduce_mean(self.ls_latent_mu, axis=0)
                    self.ls_average_latent_var = tf.reduce_mean(self.ls_latent_var, axis=0)
            else:
                with tf.variable_scope("LatentLearner", reuse=reuse):
                    self.ls_encoder_input = tf.concat([self.all_route_input, action_label, tf.stop_gradient(self.dsc_min_route_diff)], axis=1)

                    self.ls_enc_h1 = Bayesian_FC(self.ls_encoder_input, state_len + action_len * route_len + action_len + nextstate_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_enc_h1")
                    self.ls_enc_h2 = Bayesian_FC(self.ls_enc_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_enc_h2")
                    self.ls_latent = FC(self.ls_enc_h2.layer_output, 128, latent_len * 2, input_dropout = None, 
                                        output_nonln = None, name="ls_latent")
                    self.ls_latent_mu, self.ls_latent_logsig = tf.split(self.ls_latent.layer_output, 2, axis=1)
                    self.ls_latent_sig = tf.exp(tf.clip_by_value(self.ls_latent_logsig, -10, 2) - 2.)
                    noise = tf.random.normal(tf.shape(self.ls_latent_mu))

                    self.ls_output_latent = self.ls_latent_mu + self.ls_latent_sig * noise
                    self.ls_latent_var = self.ls_latent_sig ** 2
                    
                    self.ls_decoder_input = tf.concat([self.all_route_input, action_label, self.ls_output_latent], axis=1)
                    self.ls_dec_h1 = Bayesian_FC(self.ls_decoder_input, state_len + action_len * route_len + action_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1")
                    self.ls_dec_h2 = Bayesian_FC(self.ls_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2")
                    self.ls_diff = Bayesian_FC(self.ls_dec_h2.layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff")
                   
                    self.ls_average_latent_mean = tf.reduce_mean(self.ls_latent_mu, axis=0)
                    self.ls_average_latent_var = tf.reduce_mean(self.ls_latent_var, axis=0)

                    infer_route_input = tf.reshape(tf.tile(self.all_route_input, [1, action_len]), [-1, state_len + action_len * route_len])
                    infer_input_latent = tf.reshape(tf.tile(self.layer_input_latent, [1, action_len]), [-1, latent_len])
                    infer_action_label = tf.reshape(tf.tile(tf.one_hot( list(range(action_len)), action_len), [tf.shape(self.all_route_input)[0], 1]), [-1, action_len])

                    self.ls_infer_decoder_input = tf.concat([infer_route_input, infer_action_label, infer_input_latent], axis=1)
                    self.ls_infer_dec_h1 = Bayesian_FC(self.ls_infer_decoder_input, state_len + action_len * route_len + action_len + latent_len, 256, 
                                            input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="ls_dec_h1", reuse=True)
                    self.ls_infer_dec_h2 = Bayesian_FC(self.ls_infer_dec_h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="ls_dec_h2", reuse=True)
                    self.ls_infer_diff = Bayesian_FC(self.ls_infer_dec_h2.layer_output, 128, nextstate_len, input_dropout = None, 
                                        output_nonln = None, name="ls_diff", reuse=True)
                    

                    self.output_route_mean = self.dsc_output_route - tf.reshape(self.ls_infer_diff.layer_mean, [-1, action_len, nextstate_len])
                    self.output_route_var = tf.reshape(self.ls_infer_diff.layer_var, [-1, action_len, nextstate_len])


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
        self.log_latent_rec1 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec2 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec3 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec4 = np.array([0.] * self.nextstate_len)
        self.log_latent_reg = 0.
        self.log_latent_div = 0.
        self.log_disc_rec = np.array([0.] * self.nextstate_len)
        self.log_disc_action_error = np.array([0.] * self.action_len)
        self.log_disc_action_mean = np.array([0.] * self.action_len)
        self.log_num = 0

        sess = tf.get_default_session()
        sess.run(self.dsc_copy_ds)


    def network_update(self):
        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_var = np.array([0.] * self.latent_len)
        self.log_latent_rec1 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec2 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec3 = np.array([0.] * self.nextstate_len)
        self.log_latent_rec4 = np.array([0.] * self.nextstate_len)
        self.log_latent_reg = 0.
        self.log_latent_div = 0.
        self.log_disc_rec = np.array([0.] * self.nextstate_len)
        self.log_disc_action_error = np.array([0.] * self.action_len)
        self.log_disc_action_mean = np.array([0.] * self.action_len)
        self.log_num = 0


                     
    def optimize(self, input_state, input_nextstate, input_route, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_action : input_action, self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, _, l6, l7, l8 =\
                sess.run([self.ds_train_route, self.ds_train_action,
                          self.ds_min_route_error, self.ds_average_action, self.ds_action_route_error],input_list)
        
        _, _, l1, l2, l3, l4, l10, l11, l5, l9 =\
                sess.run([self.ls_train_latent_enc, self.ls_train_latent_dec, 
                          self.ls_average_latent_mean, self.ls_average_latent_var, 
                          self.ls_route_loss1, self.ls_route_loss2, self.ls_route_loss3, self.ls_route_loss4, 
                          self.ls_reg_loss, self.ls_div_loss],input_list)
        
        
        self.log_latent_mean += l1
        self.log_latent_var += l2
        self.log_latent_rec1 += l3
        self.log_latent_rec2 += l4
        self.log_latent_rec3 += l10
        self.log_latent_rec4 += l11
        self.log_latent_reg += l5
        self.log_latent_div += l9
        self.log_disc_rec += l6
        self.log_disc_action_error += l8
        self.log_disc_action_mean += l7
        self.log_num += 1

    def get_latent(self, input_state, input_nextstate, input_route, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_action : input_action, self.layer_input_dropout : 0.}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.ls_latent_mu, self.ls_latent_var], input_list)
        return l1, l2
    
    def get_output(self, input_state, input_route, input_latent, input_dropout=0.1):
        input_list = {self.layer_input_state : input_state, self.layer_input_route : input_route, self.layer_input_latent : input_latent,
                      self.layer_input_dropout : input_dropout}
        sess = tf.get_default_session()
        l1, l2, l3 = sess.run([self.dsc_output_action, self.output_route_mean, self.output_route_var], input_list)
        return l1, l2, l3

    def optimize_update(self):
        sess = tf.get_default_session()
        sess.run(self.dsc_add_ds)

    def log_caption(self):
        return "\t" + self.name + "_LatentMean\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentVar\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentReconLoss1\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_LatentReconLoss2\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_LatentReconLoss3\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_LatentReconLoss4\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_LatentRegLoss\t"  \
            + self.name + "_LatentDivLoss\t"  \
            + self.name + "_DiscReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_AverageRouteError\t" + "\t".join([ "" for _ in range(self.action_len)])  \
            + self.name + "_AverageAction\t" + "\t".join([ "" for _ in range(self.action_len)])

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_latent_mean[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_var[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_rec1[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_latent_rec2[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_latent_rec3[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_latent_rec4[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_latent_reg / log_num) + "\t"\
            + str(self.log_latent_div / log_num) + "\t"\
            + "\t".join([str(self.log_disc_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + "\t".join([str(self.log_disc_action_error[i] / log_num) for i in range(self.action_len)]) + "\t"\
            + "\t".join([str(self.log_disc_action_mean[i] / log_num) for i in range(self.action_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tLatentMean          : " + " ".join([str(self.log_latent_mean[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLatentVar           : " + " ".join([str(self.log_latent_var[i] / log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLatentReconLoss1     : " + " ".join([str(self.log_latent_rec1[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tLatentReconLoss2     : " + " ".join([str(self.log_latent_rec2[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tLatentReconLoss3     : " + " ".join([str(self.log_latent_rec3[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tLatentReconLoss4     : " + " ".join([str(self.log_latent_rec4[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tLatentRegLoss       : " + str(self.log_latent_reg / log_num) \
            + "\n\tLatentDivLoss       : " + str(self.log_latent_div / log_num) \
            + "\n\tDiscReconLoss       : " + " ".join([str(self.log_disc_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tAverageRouteError   : " + " ".join([str(self.log_disc_action_error[i] / log_num) for i in range(self.action_len)]) \
            + "\n\tAverageAction       : " + " ".join([str(self.log_disc_action_mean[i] / log_num) for i in range(self.action_len)]) )
        
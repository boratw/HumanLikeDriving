import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC, FC, Variational_FC


class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, prevstate_len = 2, nextstate_len = 2, action_len=31, latent_len=4, regularizer_weight= 0.0001,
                 learner_lr = 0.001,  sigma_bias=-1.0, isTraining=True, state_feature=128):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.state_len = state_len
        self.prevstate_len = prevstate_len
        self.nextstate_len = nextstate_len
        self.latent_len = latent_len

        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_prevstate = tf.placeholder(tf.float32, [None, prevstate_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_action = tf.placeholder(tf.int32, [None])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)

            self.state_normalizer_mean = tf.get_variable("state_normalizer_mean", dtype=tf.float32, 
                initializer=tf.zeros([state_len]), trainable=True)
            self.state_normalizer_std = tf.get_variable("state_normalizer_std", dtype=tf.float32, 
                initializer=tf.ones([state_len]), trainable=True)
            self.default_mask = tf.get_variable("default_mask", dtype=tf.float32, 
                initializer=tf.zeros([state_len]), trainable=True)
            
            self.state_normalized = (self.layer_input_state - tf.stop_gradient(self.state_normalizer_mean)) / tf.stop_gradient(self.state_normalizer_std)
            self.default_mask_normalized = tf.exp(tf.clip_by_value(self.default_mask, -5.0, 5.0))

            mask_input = tf.concat([self.state_normalized, self.layer_input_prevstate], axis=1)
            self.mask_enc1 = FC(mask_input, state_len + prevstate_len, 64, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="mask_enc1")
            self.mask_enc2 = Variational_FC(self.mask_enc1.layer_output, 64, latent_len, input_dropout = None, 
                                    output_nonln = None, name="mask_enc2")
            self.mask_latent = self.mask_enc2.layer_output

            self.mask_dec1 = FC(self.mask_latent, latent_len, 64, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="mask_dec1")
            self.mask_dec2 = FC(self.mask_dec1.layer_output, 64, state_len, input_dropout = None, 
                                    output_nonln = None, name="mask_dec2")
            
            self.mask_output_raw = self.mask_dec2.layer_output
            self.mask_output = tf.exp(tf.clip_by_value(self.mask_output_raw, -5.0, 5.0))
            
            self.state_i1 = tf.unstack(tf.stack([self.state_normalized, self.state_normalized ** 2], axis=-1), axis=1)
            self.state_i2 = [ FC(self.state_i1[i], 2, 64, input_dropout = self.layer_input_dropout, output_nonln = tf.nn.leaky_relu, name="state_i2_" + str(i))
                                for i in range(state_len) ] 
            self.state_i3 = [ FC(self.state_i2[i].layer_output, 64, state_feature, input_dropout = None, output_nonln = None, name="state_i3_" + str(i))
                                for i in range(state_len) ] 
            self.state_i4 = tf.stack( [self.state_i3[i].layer_output for i in range(state_len) ], axis=1)

            route_input_action = tf.reshape(tf.cast(self.layer_input_action, tf.float32), [-1, 1]) * 2. / (action_len - 1) - 1.

            ## Default Mask
            self.state_output = tf.reduce_sum(self.state_i4 * tf.reshape(self.default_mask_normalized, [1, -1, 1]), axis=1) / state_len
            
            self.action_h1 = FC(self.state_output, state_feature, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="action_h1")
            self.action_h2 = FC(self.action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_output")
            self.action_output_raw = tf.math.softplus(self.action_h2.layer_output)
            self.action_output = self.action_output_raw / tf.reduce_sum(self.action_output_raw, axis=1, keepdims=True)

            action_label =  tf.one_hot(self.layer_input_action, action_len)
                            
            route_input = tf.concat([self.state_output, route_input_action], axis=1)
            self.route_h1 = FC(route_input, state_feature + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h1")
            self.route_h2 = FC(self.route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h2")
            self.route_h3 = FC(self.route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3")
            self.route_output = self.route_h3.layer_output

            ## Independent
            self.ind_state_output = tf.reshape(self.state_i4, [-1, state_feature])

            self.ind_action_h1 = FC(self.ind_state_output, state_feature, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="action_h1", reuse=True)
            self.ind_action_h2 = FC(self.ind_action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_output", reuse=True)
            self.ind_action_output_raw = tf.math.softplus(self.ind_action_h2.layer_output)
            self.ind_action_output = self.ind_action_output_raw / tf.reduce_sum(self.ind_action_output_raw, axis=1, keepdims=True)

            ind_route_input_action = tf.reshape(tf.tile(route_input_action, [1, state_len]), [-1, 1])
            ind_route_input = tf.concat([self.ind_state_output, ind_route_input_action], axis=1)
            self.ind_route_h1 = FC(ind_route_input, state_feature + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h1", reuse=True)
            self.ind_route_h2 = FC(self.ind_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h2", reuse=True)
            self.ind_route_h3 = FC(self.ind_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3", reuse=True)
            self.ind_route_output = self.ind_route_h3.layer_output

            ## masked
            self.masked_state_output_i = self.state_i4 * tf.reshape(self.mask_output, [-1, state_len, 1]) * tf.reshape(self.default_mask_normalized, [1, -1, 1]) 
            self.masked_state_output = tf.reduce_sum(self.masked_state_output_i, axis=1) / state_len

            self.masked_action_h1 = FC(self.masked_state_output, state_feature, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="action_h1", reuse=True)
            self.masked_action_h2 = FC(self.masked_action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_output", reuse=True)
            self.masked_action_output_raw = tf.math.softplus(self.masked_action_h2.layer_output)
            self.masked_action_output = self.masked_action_output_raw / tf.reduce_sum(self.masked_action_output_raw, axis=1, keepdims=True)

            masked_route_input = tf.concat([self.masked_state_output, route_input_action], axis=1)
            self.masked_route_h1 = FC(masked_route_input, state_feature + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h1", reuse=True)
            self.masked_route_h2 = FC(self.masked_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.leaky_relu, name="route_h2", reuse=True)
            self.masked_route_h3 = FC(self.masked_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3", reuse=True)
            self.masked_route_output = self.masked_route_h3.layer_output



            self.mask_mean = tf.reduce_mean(self.mask_output, axis=0)
            self.mask_std = tf.math.reduce_std(self.mask_output, axis=0)
            self.latent_mean = tf.reduce_mean(self.mask_latent, axis=0)
            self.latent_std = tf.math.reduce_std(self.mask_latent, axis=0)
            
            if isTraining:
                action_label =  tf.one_hot(self.layer_input_action, action_len)

                self.default_mask_reg_loss = tf.reduce_sum(self.default_mask ** 2)
                
                self.mask_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                                [self.mask_enc1, self.mask_enc2, self.mask_dec1, self.mask_dec2]])
                
                self.state_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                                [*self.state_i2, *self.state_i3]])
                
                self.action_error = tf.reduce_mean(tf.reduce_sum(-action_label * self.action_output_raw, axis=1))
                self.action_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                                [self.action_h1, self.action_h2]])

                self.route_error = tf.reduce_mean(tf.abs(self.route_output - self.layer_input_nextstate), axis=0)
                self.route_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                                [self.route_h1, self.route_h2, self.route_h3]])
                
                ind_action_label = tf.reshape(tf.tile(action_label, [1, state_len]), [-1, action_len])
                self.ind_action_error = tf.reduce_mean(tf.reduce_sum(-ind_action_label * self.ind_action_output_raw, axis=1))
                ind_input_nextstate = tf.reshape(tf.tile(self.layer_input_nextstate, [1, state_len]), [-1, nextstate_len])
                self.ind_route_error = tf.reduce_mean(tf.abs(self.ind_route_output - ind_input_nextstate), axis=0)
                
                self.masked_action_error = tf.reduce_mean(tf.reduce_sum(-action_label * self.masked_action_output_raw, axis=1))
                self.masked_route_error = tf.reduce_mean(tf.abs(self.masked_route_output - self.layer_input_nextstate), axis=0)
                
                self.predict_optimizer = tf.train.AdamOptimizer(learner_lr)
                predict_train_vars = [self.default_mask, *self.action_h1.trainable_params, *self.action_h2.trainable_params,
                              *self.route_h1.trainable_params, *self.route_h2.trainable_params, *self.route_h3.trainable_params ]
                                                      
                self.predict_train = self.predict_optimizer.minimize(loss = tf.reduce_mean(self.route_error) + self.action_error
                                                                + (self.route_reg_loss + self.action_reg_loss + self.default_mask_reg_loss) * regularizer_weight, 
                                                                var_list = predict_train_vars)

                self.feature_optimizer = tf.train.AdamOptimizer(learner_lr)
                feature_train_vars = []
                for i in range(state_len):
                    feature_train_vars.extend(self.state_i2[i].trainable_params)
                    feature_train_vars.extend(self.state_i3[i].trainable_params)
                self.feature_train = self.feature_optimizer.minimize(loss = tf.reduce_mean(self.ind_route_error) + self.ind_action_error
                                                                + self.state_reg_loss  * regularizer_weight, 
                                                                var_list = feature_train_vars)
                

                self.mask_optimizer = tf.train.AdamOptimizer(learner_lr)
                mask_train_vars = [*self.mask_enc1.trainable_params, *self.mask_enc2.trainable_params, 
                                   *self.mask_dec1.trainable_params, *self.mask_dec2.trainable_params ]
                self.mask_train = self.mask_optimizer.minimize(loss =  tf.reduce_mean(self.masked_route_error) + self.masked_action_error
                                                                + self.mask_reg_loss * regularizer_weight, 
                                                                var_list = mask_train_vars)

                

                self.norm_optimizer = tf.train.AdamOptimizer(0.001)
                self.norm_train = self.norm_optimizer.minimize( (tf.reduce_mean(self.layer_input_state, axis=0) - self.state_normalizer_mean) ** 2
                                                              + (tf.math.reduce_std(self.layer_input_state, axis=0) - self.state_normalizer_std) ** 2 )
                    
                
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_norm_mean = np.array([0.] * self.state_len)
        self.log_norm_std = np.array([0.] * self.state_len)

        self.log_route_rec = np.array([0.] * self.nextstate_len)
        self.log_action_rec = 0.
        self.log_ind_route_rec = np.array([0.] * self.nextstate_len)
        self.log_ind_action_rec = 0.
        self.log_mask_route_rec = np.array([0.] * self.nextstate_len)
        self.log_mask_action_rec = 0.

        self.log_default_mask = np.array([0.] * self.state_len)

        self.log_mask_mean = np.array([0.] * self.state_len)
        self.log_mask_std = np.array([0.] * self.state_len)

        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_std = np.array([0.] * self.latent_len)

        self.log_mask_reg = 0.
        self.log_state_reg = 0.
        self.log_route_reg = 0.
        self.log_action_reg = 0.

        self.log_num = 0



    def network_update(self):
        self.log_norm_mean = np.array([0.] * self.state_len)
        self.log_norm_std = np.array([0.] * self.state_len)

        self.log_route_rec = np.array([0.] * self.nextstate_len)
        self.log_action_rec = 0.
        self.log_ind_route_rec = np.array([0.] * self.nextstate_len)
        self.log_ind_action_rec = 0.
        self.log_mask_route_rec = np.array([0.] * self.nextstate_len)
        self.log_mask_action_rec = 0.

        self.log_default_mask = np.array([0.] * self.state_len)

        self.log_mask_mean = np.array([0.] * self.state_len)
        self.log_mask_std = np.array([0.] * self.state_len)

        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_std = np.array([0.] * self.latent_len)

        self.log_mask_reg = 0.
        self.log_state_reg = 0.
        self.log_route_reg = 0.
        self.log_action_reg = 0.

        self.log_num = 0

                     
    def optimize(self, input_state, input_prevstate, input_nextstate, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_prevstate: input_prevstate, self.layer_input_nextstate: input_nextstate, 
                      self.layer_input_action : input_action, self.layer_input_dropout : 0.05}
        sess = tf.get_default_session()
        _, lm3, le1, le2, lr3, lr4 = sess.run([self.predict_train,
                                    self.default_mask_normalized, self.action_error, self.route_error, 
                                    self.action_reg_loss, self.route_reg_loss],input_list)
        _, le5, le6, lr2 = sess.run([self.feature_train,
                                    self.ind_action_error, self.ind_route_error,
                                    self.state_reg_loss],input_list)
        _, le3, le4, ll1, ll2, lm1, lm2, lr1 = sess.run([self.mask_train,
                                    self.masked_action_error, self.masked_route_error, 
                                    self.latent_mean, self.latent_std,
                                    self.mask_mean, self.mask_std, self.mask_reg_loss],input_list)
        _, ln1, ln2 = sess.run([self.norm_train, 
                                    self.state_normalizer_mean, self.state_normalizer_std],input_list)

        
        self.log_norm_mean += ln1
        self.log_norm_std += ln2
        self.log_action_rec += le1
        self.log_route_rec += le2
        self.log_ind_action_rec += le5
        self.log_ind_route_rec += le6
        self.log_mask_action_rec += le3
        self.log_mask_route_rec += le4
        self.log_default_mask += lm3
        self.log_mask_mean += lm1
        self.log_mask_std += lm2
        self.log_latent_mean += ll1
        self.log_latent_std += ll2
        self.log_mask_reg += lr1
        self.log_state_reg += lr2
        self.log_route_reg += lr3
        self.log_action_reg += lr4

        self.log_num += 1

    def get_output(self, input_state, input_action, discrete=True):
        input_list = {self.layer_input_state : input_state, self.layer_input_action : input_action,
                      self.layer_input_dropout : (0.0 if discrete else 0.1)}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.route_output, self.action_output] , input_list)
        return l1, l2
    

    def log_caption(self):
        return "\t" \
            + self.name + "_RouteReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_ActionReconLoss\t" \
            + self.name + "_IndRouteReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_IndActionReconLoss\t" \
            + self.name + "_MaskRouteReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_MaskActionReconLoss\t" \
            + self.name + "_NormalizerMean\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_NormalizerStd\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_DefaultMask\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_MaskMean\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_MaskStd\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_LatentMean\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentStd\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_MaskRegLoss\t" \
            + self.name + "_StateRegLoss\t" \
            + self.name + "_RouteRegLoss\t" \
            + self.name + "_ActionRegLoss\t"

        
    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" \
            + "\t".join([str(self.log_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_action_rec / log_num) + "\t"\
            + "\t".join([str(self.log_ind_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_ind_action_rec / log_num) + "\t"\
            + "\t".join([str(self.log_mask_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_mask_action_rec / log_num) + "\t"\
            + "\t".join([str(self.log_default_mask[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_norm_mean[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_norm_std[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_mask_mean[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_mask_std[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_latent_mean[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_std[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + str(self.log_mask_reg / log_num) + "\t"\
            + str(self.log_state_reg / log_num) + "\t"\
            + str(self.log_route_reg / log_num) + "\t"\
            + str(self.log_action_reg / log_num) + "\t"
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tRouteReconLoss       : " + " ".join([str(self.log_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)])\
            + "\n\tActionReconLoss      : "  + str(self.log_action_rec / log_num) \
            + "\n\tIndRouteReconLoss    : " + " ".join([str(self.log_ind_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)])\
            + "\n\tIndActionReconLoss   : "  + str(self.log_ind_action_rec / log_num) \
            + "\n\tMaskRouteReconLoss   : " + " ".join([str(self.log_mask_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)])\
            + "\n\tMaskActionReconLoss  : "  + str(self.log_mask_action_rec / log_num) \
            + "\n\tDefaultMask          : " + " ".join([str(self.log_default_mask[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tNormalizerMean       : " + " ".join([str(self.log_norm_mean[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tNormalizerStd        : " + " ".join([str(self.log_norm_std[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tMaskMean             : " + " ".join([str(self.log_mask_mean[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tMaskStd              : " + " ".join([str(self.log_mask_std[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tLatentMean           : " + " ".join([str(self.log_latent_mean[i] / log_num)[:8] for i in range(self.latent_len)])\
            + "\n\tLatentStd            : " + " ".join([str(self.log_latent_std[i] / log_num)[:8] for i in range(self.latent_len)])\
            + "\n\tMaskRegLoss          : " + str(self.log_mask_reg / log_num)\
            + "\n\tStateRegLoss         : " + str(self.log_state_reg / log_num)\
            + "\n\tRouteRegLoss         : " + str(self.log_route_reg / log_num)\
            + "\n\tActionRegLoss        : " + str(self.log_action_reg / log_num))
        
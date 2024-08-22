import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC, FC, Variational_FC


class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, prevstate_len = 2, nextstate_len = 2, param_len = 6, action_len=31, latent_len=4, regularizer_weight= 0.001,
                 learner_lr = 0.001, normalizer_lr=0.01, sigma_bias=-1.0, isTraining=True, route_loss_weight=None, action_loss_weight=1.0):

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
            self.layer_input_param = tf.placeholder(tf.float32, [None, param_len])
            self.layer_input_action = tf.placeholder(tf.int32, [None])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)
            self.layer_input_latent = tf.placeholder(tf.float32, [None, latent_len])
            self.layer_input_iter = tf.placeholder(tf.int32, [])

            self.state_normalizer_mean = tf.get_variable("state_normalizer_mean", dtype=tf.float32, 
                initializer=tf.zeros([state_len]), trainable=True)
            self.state_normalizer_std = tf.get_variable("state_normalizer_std", dtype=tf.float32, 
                initializer=tf.ones([state_len]), trainable=True)
            
            self.state_normalized = (self.layer_input_state - tf.stop_gradient(self.state_normalizer_mean)) / tf.stop_gradient(self.state_normalizer_std)
            # encoder
            encoder_input = tf.concat([self.state_normalized, self.layer_input_nextstate], axis=1)
            self.encoder_enc1 = FC(encoder_input, state_len + nextstate_len, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="encoder_enc1")
            self.encoder_enc2 = FC(self.encoder_enc1.layer_output, 128, 64, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="encoder_enc2")
            self.encoder_enc3 = Variational_FC(self.encoder_enc2.layer_output, 64, latent_len, input_dropout = None, 
                                    output_nonln = None, name="encoder_enc3")
            self.encoder_latent = self.encoder_enc3.layer_output

            # prior encoder
            encoder_prior_input = tf.concat([self.state_normalized, self.layer_input_param, self.layer_input_nextstate], axis=1)
            self.encoder_prior_enc1 = FC(encoder_prior_input, state_len + param_len + nextstate_len, 128, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="encoder_prior_enc1")
            self.encoder_prior_enc2 = FC(self.encoder_prior_enc1.layer_output, 128, 64, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="encoder_prior_enc2")
            self.encoder_prior_enc3 = Variational_FC(self.encoder_prior_enc2.layer_output, 64, latent_len, input_dropout = None, 
                                    output_nonln = None, name="encoder_prior_enc3")
            self.encoder_prior_latent = self.encoder_prior_enc3.layer_output


            route_input_action = tf.reshape(tf.cast(self.layer_input_action, tf.float32), [-1, 1]) * 2. / (action_len - 1) - 1.
            action_label =  tf.one_hot(self.layer_input_action, action_len)

            ## prior latent
            self.decoder_prior_input = tf.concat([self.state_normalized, self.encoder_prior_latent], axis=1)
            
            self.prior_action_h1 = FC(self.decoder_prior_input, state_len + latent_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="action_h1")
            self.prior_action_h2 = FC(self.prior_action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_h2")
            self.prior_action_output_raw = self.prior_action_h2.layer_output
            self.prior_action_output = tf.nn.softmax(self.prior_action_output_raw)

                            
            prior_route_input = tf.concat([self.state_normalized, self.encoder_prior_latent, route_input_action], axis=1)
            self.prior_route_h1 = FC(prior_route_input, state_len + latent_len + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h1")
            self.prior_route_h2 = FC(self.prior_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h2")
            self.prior_route_h3 = FC(self.prior_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3")
            self.prior_route_output = self.prior_route_h3.layer_output


            ## encoder latent
            encoder_action_input = tf.concat([self.state_normalized, self.encoder_latent], axis=1)
            
            self.encoder_action_h1 = FC(encoder_action_input, state_len + latent_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="action_h1", reuse=True)
            self.encoder_action_h2 = FC(self.encoder_action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_h2", reuse=True)
            self.encoder_action_output_raw = self.encoder_action_h2.layer_output
            self.encoder_action_output = tf.nn.softmax(self.encoder_action_output_raw)
                       
                            
            encoder_route_input = tf.concat([self.state_normalized, self.encoder_latent, route_input_action], axis=1)
            self.encoder_route_h1 = FC(encoder_route_input, state_len + latent_len + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h1", reuse=True)
            self.encoder_route_h2 = FC(self.encoder_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h2", reuse=True)
            self.encoder_route_h3 = FC(self.encoder_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3", reuse=True)
            self.encoder_route_output = self.encoder_route_h3.layer_output

            action_constant = tf.constant(list(range(action_len)) , dtype=tf.float32, shape=[action_len, 1])  * 2. / (action_len - 1) - 1.
            action_input = tf.tile(action_constant, [tf.shape(encoder_action_input)[0], 1])
            route_input = tf.reshape(tf.tile(encoder_action_input, [1, action_len]), [-1,  state_len + latent_len])
            test_route_input = tf.concat([route_input, action_input], axis=1)
            self.test_route_h1 = FC(test_route_input, state_len + latent_len + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h1", reuse=True)
            self.test_route_h2 = FC(self.test_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h2", reuse=True)
            self.test_route_h3 = FC(self.test_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3", reuse=True)
            self.test_route_output = tf.reshape(self.test_route_h3.layer_output, [-1, action_len, nextstate_len])

            ## input latent
            inputlatent_action_input = tf.concat([self.state_normalized, self.layer_input_latent], axis=1)
            
            self.inputlatent_action_h1 = FC(inputlatent_action_input, state_len + latent_len, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.relu, name="action_h1", reuse=True)
            self.inputlatent_action_h2 = FC(self.inputlatent_action_h1.layer_output, 256, action_len, input_dropout = None, 
                                    output_nonln = None, name="action_h2", reuse=True)
            self.inputlatent_action_output_raw = self.inputlatent_action_h2.layer_output
            self.inputlatent_action_output = tf.nn.softmax(self.inputlatent_action_output_raw)

                            
            action_constant = tf.constant(list(range(action_len)) , dtype=tf.float32, shape=[action_len, 1])  * 2. / (action_len - 1) - 1.
            action_input = tf.tile(action_constant, [tf.shape(inputlatent_action_input)[0], 1])
            route_input = tf.reshape(tf.tile(inputlatent_action_input, [1, action_len]), [-1,  state_len + latent_len])
            inputlatent_route_input = tf.concat([route_input, action_input], axis=1)
            self.inputlatent_route_h1 = FC(inputlatent_route_input, state_len + latent_len + 1, 512, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h1", reuse=True)
            self.inputlatent_route_h2 = FC(self.inputlatent_route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                output_nonln = tf.nn.relu, name="route_h2", reuse=True)
            self.inputlatent_route_h3 = FC(self.inputlatent_route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                output_nonln = None, name="route_h3", reuse=True)
            self.inputlatent_route_output = tf.reshape(self.inputlatent_route_h3.layer_output, [-1, action_len, nextstate_len])


            self.encoder_reg_loss = tf.reduce_sum([ self.encoder_enc3.regularization_loss for g in 
                                            [self.encoder_enc1, self.encoder_enc2, self.encoder_enc3]])
            self.encoder_prior_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                            [self.encoder_prior_enc1, self.encoder_prior_enc2, self.encoder_prior_enc3]])


            self.prior_action_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(action_label, self.prior_action_output_raw))
            self.prior_action_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                            [self.prior_action_h1, self.prior_action_h2]])

            self.prior_route_error = tf.reduce_mean(tf.abs(self.prior_route_output - self.layer_input_nextstate), axis=0)
            self.prior_route_reg_loss = tf.reduce_sum([ g.regularization_loss for g in 
                                            [self.prior_route_h1, self.prior_route_h2, self.prior_route_h3]])
            
            self.encoder_action_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(action_label, self.encoder_action_output_raw))
            self.encoder_route_error = tf.reduce_mean(tf.abs(self.encoder_route_output - self.layer_input_nextstate), axis=0)

            predictor_lr = tf.train.exponential_decay(learner_lr, self.layer_input_iter, 100, 0.5)
            self.predict_optimizer = tf.train.AdamOptimizer(predictor_lr)
            predict_train_vars = [*self.encoder_prior_enc1.trainable_params, *self.encoder_prior_enc2.trainable_params, *self.encoder_prior_enc3.trainable_params,
                            *self.prior_action_h1.trainable_params, *self.prior_action_h2.trainable_params,
                            *self.prior_route_h1.trainable_params, *self.prior_route_h2.trainable_params, *self.prior_route_h3.trainable_params ]
                                                    
            self.predict_train = self.predict_optimizer.minimize(loss = 
                                    tf.reduce_mean(self.prior_route_error * (route_loss_weight if route_loss_weight != None else 1.0)) 
                                    + self.prior_action_error * action_loss_weight
                                    + (self.prior_route_reg_loss + self.prior_action_reg_loss + self.encoder_prior_reg_loss) * regularizer_weight, 
                                    var_list = predict_train_vars)
            

            latent_lr = tf.train.exponential_decay(learner_lr, self.layer_input_iter, 100, 0.9)
            self.latent_optimizer = tf.train.AdamOptimizer(latent_lr)
            latent_train_vars = [*self.encoder_enc1.trainable_params, *self.encoder_enc2.trainable_params, *self.encoder_enc3.trainable_params]
            self.latent_error = tf.reduce_mean((self.encoder_enc3.mu - self.encoder_prior_enc3.mu) ** 2)
            self.latent_train = self.latent_optimizer.minimize(loss = 
                                    tf.reduce_mean(self.encoder_route_error * (route_loss_weight if route_loss_weight != None else 1.0)) 
                                    + self.encoder_action_error * action_loss_weight + self.latent_error
                                    + self.encoder_reg_loss * regularizer_weight, 
                                    var_list = latent_train_vars)

            
            normalizer_lr = tf.train.exponential_decay(normalizer_lr, self.layer_input_iter, 20, 0.1)
            self.norm_optimizer = tf.train.AdamOptimizer(normalizer_lr)
            self.norm_train = self.norm_optimizer.minimize( (tf.reduce_mean(self.layer_input_state, axis=0) - self.state_normalizer_mean) ** 2
                                                            + (tf.math.reduce_std(self.layer_input_state, axis=0) - self.state_normalizer_std) ** 2 )
        


            self.latent_mean = tf.reduce_mean(self.encoder_latent, axis=0)
            self.latent_std = tf.math.reduce_std(self.encoder_latent, axis=0)
                
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

        self.log_prior_route_rec = np.array([0.] * self.nextstate_len)
        self.log_prior_action_rec = 0.

        self.log_encoder_route_rec = np.array([0.] * self.nextstate_len)
        self.log_encoder_action_rec = 0.

        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_std = np.array([0.] * self.latent_len)

        self.log_encoder_reg = 0.
        self.log_encoder_prior_reg = 0.
        self.log_route_reg = 0.
        self.log_action_reg = 0.

        self.log_latent_loss = 0.

        self.log_num = 0



    def network_update(self):
        self.log_norm_mean = np.array([0.] * self.state_len)
        self.log_norm_std = np.array([0.] * self.state_len)

        self.log_prior_route_rec = np.array([0.] * self.nextstate_len)
        self.log_prior_action_rec = 0.

        self.log_encoder_route_rec = np.array([0.] * self.nextstate_len)
        self.log_encoder_action_rec = 0.

        self.log_latent_mean = np.array([0.] * self.latent_len)
        self.log_latent_std = np.array([0.] * self.latent_len)

        self.log_encoder_reg = 0.
        self.log_encoder_prior_reg = 0.
        self.log_route_reg = 0.
        self.log_action_reg = 0.

        self.log_latent_loss = 0.

        self.log_num = 0

                     
    def optimize(self, input_iter, input_state, input_prevstate, input_nextstate, input_param, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_prevstate: input_prevstate, self.layer_input_nextstate: input_nextstate, 
                      self.layer_input_action : input_action, self.layer_input_param : input_param, self.layer_input_iter : input_iter,
                      self.layer_input_dropout : 0.05}
        sess = tf.get_default_session()
        _, le1, le2, lr2, lr3, lr4, ll1, ll2, = sess.run([self.predict_train,
                                    self.prior_action_error, self.prior_route_error, self.encoder_prior_reg_loss,
                                    self.prior_action_reg_loss, self.prior_route_reg_loss, self.latent_mean, self.latent_std],input_list)
        _, le3, le4, lm1, lr1 = sess.run([self.latent_train, self.encoder_action_error, self.encoder_route_error,
                                    self.latent_error, self.encoder_reg_loss],input_list)
        _, ln1, ln2 = sess.run([self.norm_train, 
                                    self.state_normalizer_mean, self.state_normalizer_std],input_list)

        
        self.log_norm_mean += ln1
        self.log_norm_std += ln2

        self.log_prior_action_rec += le1
        self.log_prior_route_rec += le2

        self.log_encoder_action_rec += le3
        self.log_encoder_route_rec += le4

        self.log_latent_mean += ll1
        self.log_latent_std += ll2
        
        self.log_encoder_reg += lr1
        self.log_encoder_prior_reg += lr2
        self.log_route_reg += lr3
        self.log_action_reg += lr4

        self.log_latent_loss += lm1

        self.log_num += 1

    def get_output(self, input_state, discrete=True):
        input_list = {self.layer_input_state : input_state, self.layer_input_dropout : (0.0 if discrete else 0.1)}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.test_route_output, self.encoder_action_output] , input_list)
        return l1, l2, None
    
    def get_output_latent(self, input_state, lnput_latent, discrete=True):
        input_list = {self.layer_input_state : input_state, self.layer_input_latent : lnput_latent, self.layer_input_dropout : (0.0 if discrete else 0.1)}
        sess = tf.get_default_session()
        l1, l2 = sess.run([self.inputlatent_route_output, self.inputlatent_action_output] , input_list)
        return l1, l2, None
    
    def get_latent(self, input_state, input_nextstate, discrete=True):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate,
                      self.layer_input_dropout : (0.0 if discrete else 0.1)}
        sess = tf.get_default_session()
        l1 = sess.run(self.encoder_latent , input_list)
        return l1
    
    

    def log_caption(self):
        return "\t" \
            + self.name + "_NormalizerMean\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_NormalizerStd\t" + "\t".join([ "" for _ in range(self.state_len)]) \
            + self.name + "_PriorRouteReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_PriorActionReconLoss\t" \
            + self.name + "_EncoderRouteReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_EncoderActionReconLoss\t" \
            + self.name + "_LatentMean\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentStd\t" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_LatentRegLoss\t" \
            + self.name + "_PriorRegLoss\t" \
            + self.name + "_RouteRegLoss\t" \
            + self.name + "_ActionRegLoss\t"\
            + self.name + "_LatentLoss\t"

        
    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" \
            + "\t".join([str(self.log_norm_mean[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_norm_std[i] / log_num) for i in range(self.state_len)]) + "\t"\
            + "\t".join([str(self.log_prior_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_prior_action_rec / log_num) + "\t"\
            + "\t".join([str(self.log_encoder_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_encoder_action_rec / log_num) + "\t"\
            + "\t".join([str(self.log_latent_mean[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + "\t".join([str(self.log_latent_std[i] / log_num) for i in range(self.latent_len)]) + "\t"\
            + str(self.log_encoder_reg / log_num) + "\t"\
            + str(self.log_encoder_prior_reg / log_num) + "\t"\
            + str(self.log_route_reg / log_num) + "\t"\
            + str(self.log_action_reg / log_num) + "\t"\
            + str(self.log_latent_loss / log_num) + "\t"
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tNormalizerMean       : " + " ".join([str(self.log_norm_mean[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tNormalizerStd        : " + " ".join([str(self.log_norm_std[i] / log_num)[:6] for i in range(self.state_len)])\
            + "\n\tPriorRouteReconLoss  : " + " ".join([str(self.log_prior_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)])\
            + "\n\tPriorActionReconLoss : "  + str(self.log_prior_action_rec / log_num) \
            + "\n\tEncRouteReconLoss    : " + " ".join([str(self.log_encoder_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)])\
            + "\n\tEncActionReconLoss   : "  + str(self.log_encoder_action_rec / log_num) \
            + "\n\tLatentMean           : " + " ".join([str(self.log_latent_mean[i] / log_num)[:8] for i in range(self.latent_len)])\
            + "\n\tLatentStd            : " + " ".join([str(self.log_latent_std[i] / log_num)[:8] for i in range(self.latent_len)])\
            + "\n\tLatentRegLoss        : " + str(self.log_encoder_reg / log_num)\
            + "\n\tPriorRegLoss         : " + str(self.log_encoder_prior_reg / log_num)\
            + "\n\tRouteRegLoss         : " + str(self.log_route_reg / log_num)\
            + "\n\tActionRegLoss        : " + str(self.log_action_reg / log_num)\
            + "\n\tLatentLoss           : " + str(self.log_latent_loss / log_num))
        
import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC, FC, Variational_FC


class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, action_len=31, regularizer_weight= 0.0001,
                 learner_lr = 0.001,  sigma_bias=-1.0, bayesian_out_regularizer = 0.01, isTraining=True):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.state_len = state_len
        self.nextstate_len = nextstate_len

        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_action = tf.placeholder(tf.int32, [None])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)

            with tf.variable_scope("DiscreteLearner", reuse=reuse):
                self.action_h1 = Bayesian_FC(self.layer_input_state, state_len, 128, input_dropout = self.layer_input_dropout, 
                                        output_nonln = tf.nn.leaky_relu, name="action_h1", sigma_bias=sigma_bias)
                self.action_h2 = FC(self.action_h1.layer_output, 128, action_len, input_dropout = None, 
                                        output_nonln = None, name="action_output")
                self.action_output_raw = self.action_h2.layer_output
                self.action_output = tf.nn.softmax(self.action_output_raw)

                action_label =  tf.one_hot(self.layer_input_action, action_len)
                
                
                    
                route_input_action = tf.reshape(tf.cast(self.layer_input_action, tf.float32), [-1, 1]) * 2. / (action_len - 1) - 1.
                route_input = tf.concat([self.layer_input_state, route_input_action], axis=1)
                self.route_h1 = Bayesian_FC(route_input, state_len + 1, 512, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="route_h1")
                self.route_h2 = Bayesian_FC(self.route_h1.layer_output, 512, 256, input_dropout = self.layer_input_dropout, 
                                    output_nonln = tf.nn.leaky_relu, name="route_h2")
                self.route_h3 = FC(self.route_h2.layer_output, 256, nextstate_len, input_dropout = None, 
                                    output_nonln = None, name="route_h3")
                self.route_output = self.route_h3.layer_output
                
                if isTraining:
                    self.action_error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(action_label, self.action_output_raw))
                    self.action_reg_loss = tf.reduce_mean([ tf.reduce_sum(g.regularization_loss) for g in 
                                                    [self.action_h1, self.action_h2]])
                    
                    self.action_optimizer = tf.train.AdamOptimizer(learner_lr)
                    self.action_train_gradient = self.action_optimizer.compute_gradients(loss = self.action_error
                                                                    + self.action_reg_loss * regularizer_weight)
                    action_train_gradient_clipped = [ ( None if g is None else tf.clip_by_value(g, -0.1, 0.1), v) for g, v in self.action_train_gradient]
                    self.action_train = self.action_optimizer.apply_gradients(action_train_gradient_clipped)


                    self.route_error = tf.reduce_mean((self.route_output - self.layer_input_nextstate) ** 2, axis=0)
                    self.route_reg_loss = tf.reduce_mean([ tf.reduce_sum(g.regularization_loss) for g in 
                                                    [self.route_h1, self.route_h2, self.route_h3]])
                    
                    self.route_optimizer = tf.train.AdamOptimizer(learner_lr)
                    self.route_train_gradient = self.route_optimizer.compute_gradients(loss = self.route_error
                                                                    + self.route_reg_loss * regularizer_weight)
                    route_train_gradient_clipped = [ ( None if g is None else tf.clip_by_value(g, -0.1, 0.1), v) for g, v in self.route_train_gradient]
                    self.route_train = self.route_optimizer.apply_gradients(route_train_gradient_clipped)

                    
                
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_route_rec = np.array([0.] * self.nextstate_len)
        self.log_route_reg = 0.
        self.log_action_rec = 0.
        self.log_action_reg = 0.
        self.log_num = 0



    def network_update(self):
        self.log_route_rec = np.array([0.] * self.nextstate_len)
        self.log_route_reg = 0.
        self.log_action_rec = 0.
        self.log_action_reg = 0.
        self.log_num = 0


                     
    def optimize(self, input_state, input_nextstate, input_action):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, 
                      self.layer_input_action : input_action, self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, l1, l2 = sess.run([self.route_train, self.route_error, self.route_reg_loss],input_list)

        _, l3, l4 = sess.run([self.action_train, self.action_error, self.action_reg_loss],input_list)
        
        self.log_route_rec += l1
        self.log_route_reg += l2
        self.log_action_rec += l3
        self.log_action_reg += l4
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
            + self.name + "_RouteRegLoss\t"\
            + self.name + "_ActionReconLoss\t"\
            + self.name + "_ActionRegLoss\t"

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" \
            + "\t".join([str(self.log_route_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_route_reg / log_num) + "\t"\
            + str(self.log_action_rec / log_num) + "\t"\
            + str(self.log_action_reg / log_num) + "\t"
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tRouteReconLoss       : " + " ".join([str(self.log_route_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tRouteRegLoss         : " + str(self.log_route_reg / log_num)\
            + "\n\tActionReconLoss      : " + str(self.log_action_rec / log_num)\
            + "\n\tActionRegLoss        : " + str(self.log_action_reg / log_num))
        
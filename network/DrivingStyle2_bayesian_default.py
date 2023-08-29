import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, regularizer_weight= 0.001,
                 lr = 0.001, not_action_lr = 0.05):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.nextstate_len = nextstate_len
        self.route_len = route_len
        self.action_len = action_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_route = tf.placeholder(tf.float32, [None, action_len, route_len])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)
            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])

            self.all_route_input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            self.h1 = Bayesian_FC(self.all_route_input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="h1")
            self.l_action = Bayesian_FC(self.h1.layer_output, 256, action_len, input_dropout = None, 
                                  output_nonln = None, name="l_action")
            self.action_normalizer = tf.get_variable("action_normalizer", dtype=tf.float32, 
                initializer=tf.ones([action_len]) / action_len, trainable=True)
            
            self.output_action = tf.nn.softmax(self.l_action.layer_output, axis=1)
            self.h2 = [Bayesian_FC(self.h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="h2_" + str(i)) for i in range(action_len)]
            self.l_route = [Bayesian_FC(self.h2[i].layer_output, 128, nextstate_len, input_dropout = None, 
                                  output_nonln = None, name="l_route" + str(i)) for i in range(action_len)]
            self.output_route = tf.stack([self.l_route[i].layer_output for i in range(action_len)], axis=1)

            self.route_error = tf.reduce_mean((self.output_route - tf.reshape(self.layer_input_nextstate, [-1, 1, nextstate_len])) ** 2, axis=2)
            self.minimum_loss_action = tf.math.argmin(self.route_error, axis=1)
            self.minimum_route_error = tf.reduce_mean((tf.gather(self.output_route, self.minimum_loss_action, axis=1, batch_dims=1) - self.layer_input_nextstate) ** 2, axis=0)
            self.maximum_output_action = tf.reduce_mean(tf.reduce_max(self.output_action, axis=1))
            

            self.route_reg_loss = self.h1.regularization_loss
            for i in range(action_len):
                self.route_reg_loss += self.h2[i].regularization_loss
                self.route_reg_loss += self.l_route[i].regularization_loss
            self.route_reg_loss /= (action_len * 2. + 1)
            self.action_reg_loss = (self.h1.regularization_loss + self.l_action.regularization_loss) / 2.

            self.optimizer = tf.train.AdamOptimizer(lr)
            train_route_vars = [*self.h1.trainable_params]
            for i in range(action_len):
                train_route_vars.extend(self.h2[i].trainable_params)
                train_route_vars.extend(self.l_route[i].trainable_params)
            self.train_route = self.optimizer.minimize(loss = tf.reduce_mean(self.minimum_route_error) + tf.reduce_mean(self.route_error) * not_action_lr
                                                       + self.route_reg_loss * regularizer_weight,
                                                       var_list=train_route_vars )
            action_possibility = self.action_normalizer / self.route_error
            action_possibility_softmax = action_possibility / tf.reduce_sum(action_possibility, axis=1, keepdims=True)
            self.action_error = - tf.stop_gradient(action_possibility_softmax) * self.output_action
            self.train_action = self.optimizer.minimize(loss = self.action_error + self.action_reg_loss * regularizer_weight,
                                                       var_list=[*self.h1.trainable_params, *self.l_action.trainable_params] )     

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
        self.log_rec = np.array([0.] * self.nextstate_len)
        self.log_action = 0.
        self.log_a_reg = 0.
        self.log_r_reg = 0.
        self.log_a_norm = np.array([0.] * self.action_len)
        self.log_num = 0

    def network_update(self):
        self.log_rec = np.array([0.] * self.nextstate_len)
        self.log_action = 0.
        self.log_a_reg = 0.
        self.log_r_reg = 0.
        self.log_a_norm = np.array([0.] * self.action_len)
        self.log_num = 0
            
    def optimize(self, input_state, input_nextstate, input_route):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, _, _, l1, l2, l3, l4, l5  = sess.run([self.train_route, self.train_action, self.train_normalizer, 
                                       self.minimum_route_error, self.maximum_output_action, self.action_reg_loss, self.route_reg_loss, 
                                       self.action_normalizer ],input_list)
        
        self.log_rec += l1
        self.log_action += l2
        self.log_a_reg += l3
        self.log_r_reg += l4
        self.log_a_norm += l5
        self.log_num += 1
       
    def log_caption(self):
        return "\t" + self.name + "_ReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_MaximumAction\t"  \
            + self.name + "_ActionRegLoss\t"  \
            + self.name + "_RouteRegLoss\t"  \
            + self.name + "_AverageAction\t" + "\t".join([ "" for _ in range(self.action_len)])

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.log_rec[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.log_action / log_num) + "\t"\
            + str(self.log_a_reg / log_num) + "\t"\
            + str(self.log_r_reg / log_num) + "\t"\
            + "\t".join([str(self.log_a_norm[i] / log_num) for i in range(self.action_len)])
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tReconLoss          : " + " ".join([str(self.log_rec[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tMaximumAction      : " + str(self.log_action / log_num) \
            + "\n\tActionRegLoss      : " + str(self.log_a_reg / log_num) \
            + "\n\tRouteRegLoss       : " + str(self.log_r_reg / log_num) \
            + "\n\tAverageAction      : " + " ".join([str(self.log_a_norm[i] / log_num) for i in range(self.action_len)]) )
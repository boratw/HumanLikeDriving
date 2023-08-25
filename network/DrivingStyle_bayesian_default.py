import tensorflow.compat.v1 as tf
import numpy as np
import math

from network.bayesian_mlp import Bayesian_FC



class DrivingStyleLearner():
    def __init__(self, name=None, reuse=False, state_len = 59, nextstate_len = 2, route_len = 10, action_len=3, regularizer_weight= 0.001,
                 lr = 0.001):

        if name == None:
            self.name = "DrivingStyleLearner"
        else:
            self.name = "DrivingStyleLearner" + name
        self.nextstate_len = nextstate_len
        self.route_len = route_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len])
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, nextstate_len])
            self.layer_input_route = tf.placeholder(tf.float32, [None, action_len, route_len])
            self.layer_input_dropout = tf.placeholder(tf.float32, None)
            layer_input_route_flatten = tf.reshape(self.layer_input_route, [-1, action_len * route_len])

            self.input = tf.concat([self.layer_input_state, layer_input_route_flatten], axis=1)
            self.h1 = Bayesian_FC(self.input, state_len + action_len * route_len, 256, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="h1")
            self.h2 = Bayesian_FC(self.h1.layer_output, 256, 128, input_dropout = self.layer_input_dropout, 
                                  output_nonln = tf.nn.leaky_relu, name="h2")
            self.output = Bayesian_FC(self.h2.layer_output, 128, nextstate_len, input_dropout = None, 
                                  output_nonln = None, name="output")
            
            self.error =  tf.reduce_mean((self.output.layer_output - self.layer_input_nextstate) ** 2, axis=0)
            self.regularization_loss = self.h1.regularization_loss + self.h2.regularization_loss + self.output.regularization_loss

            self.loss = tf.reduce_mean(self.error) + self.regularization_loss * regularizer_weight

            self.optimizer = tf.train.AdamOptimizer(lr)
            self.train_action = self.optimizer.minimize(loss = self.loss)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.rec_loss = np.array([0.] * self.nextstate_len)
        self.reg_loss = 0.
        self.log_num = 0

    def network_update(self):
        self.rec_loss = np.array([0.] * self.nextstate_len)
        self.reg_loss = 0.
        self.log_num = 0
            
    def optimize(self, input_state, input_nextstate, input_route):
        input_list = {self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate, self.layer_input_route : input_route,
                      self.layer_input_dropout : 0.1}
        sess = tf.get_default_session()
        _, l1, l2 = sess.run([self.train_action, self.error,  self.regularization_loss],input_list)
        
        self.rec_loss += l1
        self.reg_loss += l2
        self.log_num += 1
       
    def log_caption(self):
        return "\t" + self.name + "_ReconLoss\t" + "\t".join([ "" for _ in range(self.nextstate_len)]) \
            + self.name + "_RegLoss\t"  

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return "\t" + "\t".join([str(self.rec_loss[i] / log_num) for i in range(self.nextstate_len)]) + "\t"\
            + str(self.reg_loss / log_num)
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tReconLoss          : " + " ".join([str(self.rec_loss[i] / log_num)[:8] for i in range(self.nextstate_len)]) \
            + "\n\tRegLoss            : " + str(self.reg_loss / log_num) )
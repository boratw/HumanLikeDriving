import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP



class DriveStylePredictor():
    def __init__(self, name=None, reuse=False, learner_lr=0.0001):

        if name == None:
            self.name = "DrvStylePredictor"
        else:
            self.name = "DrvStylePredictor_" + name
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, 5])
            self.layer_input_route = tf.placeholder(tf.float32, [None, 3, 10])
            self.layer_input_target = tf.placeholder(tf.float32, [None, 7])

            layer_input = tf.concat([self.layer_input_state, tf.reshape(self.layer_input_route, [-1, 30])], axis=1)
            self.mlp = MLP(35, 7, [64, 64, 64], hidden_nonlns = tf.nn.tanh, input_tensor=layer_input, use_dropout=True)
            self.layer_dropout = self.mlp.layer_dropout
            
            self.loss = (self.mlp.layer_output - self.layer_input_target) ** 2
            #self.loss = tf.clip_by_value(self.loss, 0, 0.01)
            self.loss = tf.reduce_mean(self.loss, axis=0)


            self.optimizer = tf.train.AdamOptimizer(learner_lr)
            self.train_action = self.optimizer.minimize(loss = tf.reduce_sum(self.loss))


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_loss = np.array([0.] * 7)
        self.log_num = 0

    def network_update(self):
        self.log_loss = np.array([0.] * 7)
        self.log_num = 0
            
    def optimize_batch(self, input_state, input_route, input_target):
        input_list = {self.layer_input_state : input_state, self.layer_input_route: input_route, 
                      self.layer_input_target : input_target, self.layer_dropout : 0.1}
        sess = tf.get_default_session()
        _, l = sess.run([self.train_action, self.loss], input_list)

        self.log_loss += l
        self.log_num += 1

    def log_caption(self):
        return "\t"  + self.name + "_Loss"

    def current_log(self):
        return "".join(["\t" + str(self.log_loss[i] / self.log_num) for i in range(7)])

    def log_print(self):
        print ( self.name + "\n" \
            + "\tLearner_Loss      : " + " ".join([str(self.log_loss[i] / self.log_num) for i in range(7)]))
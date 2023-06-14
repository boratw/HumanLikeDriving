import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Encoder, VAE_Decoder
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class DrivingController():
    def __init__(self, name=None, reuse=False, state_len = 5, action_len = 2,
                 learner_lr_start = 1, learner_lr_end = 0.01, learner_lr_step = 1000, value_lr=0.001, policy_lr=0.0001, alpha_lr=0.001,
                 policy_regularizer_weight=0.001, l2_regularizer_weight=0.0001, policy_update_ratio=0.5, policy_gamma=0.9):

        if name == None:
            self.name = "DrivingStyleController"
        else:
            self.name = "DrivingStyleController" + name
        self.action_len = action_len
        self.target_entropy=-action_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, state_len], name="layer_input_state")
            self.layer_input_nextstate = tf.placeholder(tf.float32, [None, state_len], name="layer_input_nextstate")
            self.layer_input_action = tf.placeholder(tf.float32, [None, action_len], name="layer_input_action")
            self.layer_input_reward = tf.placeholder(tf.float32, [None, 1], name="layer_input_reward")


            self.layer_iteration_num = tf.placeholder(tf.int32, None)
            self.layer_dropout = tf.placeholder(tf.float32, None)

            self.log_alpha = tf.Variable(-5.3, trainable=True)
            self.alpha = tf.exp(self.log_alpha)
            self.lr = learner_lr_end + tf.exp(-self.layer_iteration_num / learner_lr_step) * (learner_lr_start - learner_lr_end)
            self.value_lr = self.lr * value_lr
            self.policy_lr = self.lr * policy_lr
            self.alpha_lr = self.lr * alpha_lr

            self.policy = VAE_Encoder(state_len, action_len, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_state, name="Policy")
            self.next_policy = VAE_Encoder(state_len, action_len, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_input_nextstate, name="Policy", reuse=True)
            
            qvalue_input = tf.concat([self.layer_input_state, self.layer_input_action], axis=1)
            self.qvalue1 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=qvalue_input, name="Qvalue1")
            self.qvalue2 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=qvalue_input, name="Qvalue2")

            qvalue_policy_input = tf.concat([self.layer_input_state, self.policy.layer_output], axis=1) 
            self.qvalue_policy1 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=qvalue_policy_input, name="Qvalue1", reuse=True)
            self.qvalue_policy2 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=qvalue_policy_input, name="Qvalue2", reuse=True)

            next_qvalue_input = tf.concat([self.layer_input_nextstate, self.next_policy.layer_output], axis=1)
            self.next_qvalue1 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=next_qvalue_input, name="Qvalue_Target1")
            self.next_qvalue2 = MLP(state_len + action_len, 1, [256, 256], hidden_nonlns = tf.nn.leaky_relu, input_tensor=next_qvalue_input, name="Qvalue_Target2")

            self.qvalue1_assign = self.next_qvalue1.build_add_weighted(self.qvalue1, 1.0)
            self.qvalue1_update = self.next_qvalue1.build_add_weighted(self.qvalue1, policy_update_ratio)
            self.qvalue2_assign = self.next_qvalue2.build_add_weighted(self.qvalue2, 1.0)
            self.qvalue2_update = self.next_qvalue2.build_add_weighted(self.qvalue2, policy_update_ratio)
        
            min_next_Q =  tf.minimum(self.next_qvalue1.layer_output, self.next_qvalue2.layer_output)
            Q_target = tf.stop_gradient(self.layer_input_reward + (min_next_Q - self.next_policy.log_pi * self.alpha) * policy_gamma)
            
            self.qvalue1_optimizer = tf.train.AdamOptimizer(self.value_lr)
            self.qvalue1_rec_loss = tf.reduce_mean((self.qvalue1.layer_output - Q_target) ** 2)
            self.qvalue1_loss = self.qvalue1_rec_loss + self.qvalue1.l2_loss * l2_regularizer_weight
            self.qvalue1_train = self.qvalue1_optimizer.minimize(self.qvalue1_loss, var_list=self.qvalue1.trainable_params)

            self.qvalue2_optimizer = tf.train.AdamOptimizer(self.value_lr)
            self.qvalue2_rec_loss = tf.reduce_mean((self.qvalue2.layer_output - Q_target) ** 2)
            self.qvalue2_loss = self.qvalue2_rec_loss + self.qvalue2.l2_loss * l2_regularizer_weight
            self.qvalue2_train = self.qvalue2_optimizer.minimize(self.qvalue2_loss, var_list=self.qvalue2.trainable_params)
            
            mean_Q = tf.reduce_mean([self.qvalue_policy1.layer_output, self.qvalue_policy2.layer_output], axis=0)
            self.policy_rec_loss = tf.reduce_mean(self.policy.log_pi * self.alpha - mean_Q) 
            self.policy_loss = self.policy_rec_loss + self.policy.regularization_loss * policy_regularizer_weight + self.policy.encoder.l2_loss * l2_regularizer_weight
            self.policy_optimizer = tf.train.AdamOptimizer(policy_lr)
            self.policy_train = self.policy_optimizer.minimize(self.policy_loss, var_list=self.policy.trainable_params)
            
            self.alpha_loss = tf.reduce_mean(-1. * (self.alpha * tf.stop_gradient(self.policy.log_pi + self.target_entropy)))
            self.alpha_optimizer = tf.train.AdamOptimizer(alpha_lr)
            self.alpha_train = self.alpha_optimizer.minimize(self.alpha_loss, var_list=[self.log_alpha])

            self.qvalue1_average = tf.reduce_mean(self.qvalue1.layer_output)
            self.qvalue2_average = tf.reduce_mean(self.qvalue2.layer_output)

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)


            
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_assign, self.qvalue2_assign])

        self.log_qvalue1_avg = 0.
        self.log_qvalue1_rec = 0.
        self.log_qvalue1_reg = 0.
        self.log_qvalue2_avg = 0.
        self.log_qvalue2_rec = 0.
        self.log_qvalue2_reg = 0.
        self.log_policy_reg = 0.
        self.log_policy_div_rec = 0.
        self.log_policy_l2_rec = 0.
        self.log_alpha = 0.
        self.log_num = 0

    def network_update(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_update, self.qvalue2_update])

        self.log_qvalue1_avg = 0.
        self.log_qvalue1_rec = 0.
        self.log_qvalue1_reg = 0.
        self.log_qvalue2_avg = 0.
        self.log_qvalue2_rec = 0.
        self.log_qvalue2_reg = 0.
        self.log_policy_reg = 0.
        self.log_policy_div_rec = 0.
        self.log_policy_l2_rec = 0.
        self.log_alpha = 0.
        self.log_num = 0
            
    def optimize(self, iteration_num, input_state, input_nextstate, input_action, input_reward):
        input_list = {self.layer_iteration_num : iteration_num, self.layer_input_state : input_state, self.layer_input_nextstate: input_nextstate,
                      self.layer_input_action : input_action, self.layer_input_reward : input_reward}
        sess = tf.get_default_session()
        _, _, l1, l2, l3, l4, l5, l6 = sess.run([self.qvalue1_train, self.qvalue2_train, self.qvalue1_average, self.qvalue2_average, 
                                                 self.qvalue1_rec_loss, self.qvalue2_rec_loss, self.qvalue1.l2_loss, self.qvalue2.l2_loss],input_list)
        
        _, _, l7, l8, l9, l10 = sess.run([self.policy_train, self.alpha_train, self.policy_rec_loss, self.policy.regularization_loss, self.policy.encoder.l2_loss,
                                                 self.alpha],input_list)
        
        self.log_qvalue1_avg += l1
        self.log_qvalue1_rec += l3
        self.log_qvalue1_reg += l5
        self.log_qvalue2_avg += l2
        self.log_qvalue2_rec += l4
        self.log_qvalue2_reg += l6
        self.log_policy_reg += l7
        self.log_policy_div_rec += l8
        self.log_policy_l2_rec += l9
        self.log_alpha += l10
        self.log_num += 1

    def get_action(self, input_state):
        input_list = {self.layer_input_state : input_state}
        sess = tf.get_default_session()
        out = sess.run(self.policy.layer_output, input_list)
        return out

       
    def log_caption(self):
        return "\t" + self.name + "_Qvalue1_Average\t" + self.name + "_Qvalue1_Loss\t" + self.name + "_Qvalue1_L2Loss\t" \
            + self.name + "_Qvalue2_Average\t" + self.name + "_Qvalue2_Loss\t" + self.name + "_Qvalue2_L2Loss\t" \
            + self.name + "_Policy_Loss\t" + self.name + "_Policy_DivLoss\t" + self.name + "_Policy_L2Loss\t" + self.name + "_Alpha\t"

    def current_log(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        return str(self.log_qvalue1_avg / log_num) + "\t" + str(self.log_qvalue1_rec / log_num) + "\t"  + str(self.log_qvalue1_reg / log_num) + "\t" \
            + str(self.log_qvalue2_avg / log_num) + "\t" + str(self.log_qvalue2_rec / log_num) + "\t"  + str(self.log_qvalue2_reg / log_num) + "\t" \
            + str(self.log_policy_reg / log_num) + "\t" + str(self.log_policy_div_rec / log_num) + "\t"  + str(self.log_policy_l2_rec / log_num) + "\t" \
            + str(self.log_alpha / log_num) 
    
    def log_print(self):
        log_num = (self.log_num if self.log_num > 0 else 1)
        print ( self.name \
            + "\n\tQvalue1_Average        : " + str(self.log_qvalue1_avg / log_num) \
            + "\n\tQvalue1_Loss           : " + str(self.log_qvalue1_rec / log_num) \
            + "\n\tQvalue1_L2Loss         : " + str(self.log_qvalue1_reg / log_num) \
            + "\n\tQvalue2_Average        : " + str(self.log_qvalue2_avg / log_num) \
            + "\n\tQvalue2_Loss           : " + str(self.log_qvalue2_rec / log_num) \
            + "\n\tQvalue2_L2Loss         : " + str(self.log_qvalue2_reg / log_num) \
            + "\n\tPolicy_Loss            : " + str(self.log_policy_reg / log_num) \
            + "\n\tPolicy_DivLoss         : " + str(self.log_policy_div_rec / log_num) \
            + "\n\tPolicy_L2Loss          : " + str(self.log_policy_l2_rec / log_num) \
            + "\n\tAlpha                  : " + str(self.log_alpha / log_num) )
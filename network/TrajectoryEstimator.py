import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE



class TrajectoryEstimator():
    def __init__(self, name=None, reuse=False, learner_lr=0.001, regularizer_weight=0.001):

        if name == None:
            self.name = "TrajectoryEstimator"
        else:
            self.name = "TrajectoryEstimator" + name
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, 2])
            self.layer_input_route = tf.placeholder(tf.float32, [None, 3, 10])
            self.layer_input_othervcs = tf.placeholder(tf.float32, [None, 8, 5])
            self.layer_input_traj = tf.placeholder(tf.float32, [None, 10, 2])

            layer_input_route_reshaped = tf.reshape(self.layer_input_route, [-1, 30])

            layer_input_othervcs = tf.unstack(self.layer_input_othervcs, axis=1)
            othervcs_embeddings = []
            for i, layer in enumerate(layer_input_othervcs):
                outlayer = MLP(5, 8, [32], hidden_nonlns = tf.nn.tanh, input_tensor=layer, name="OtherVcsEmbedding" + str(i))
                othervcs_embeddings.append(outlayer.layer_output)
            self.othervcs_embedding = tf.reduce_sum(othervcs_embeddings, axis=0)

            self.layer_input = tf.concat([self.layer_input_state, layer_input_route_reshaped, self.othervcs_embedding], axis=1)
            self.layer_output = tf.reshape(self.layer_input_traj, [-1, 20])

            self.vae = VAE(20, 8, [64, 64, 64], hidden_nonlns = tf.nn.tanh, input_tensor=self.layer_output,
                           additional_dim=40, additional_tensor=self.layer_input )

            
            self.reconstruction_loss = tf.reduce_mean((self.vae.layer_output - self.layer_output) ** 2)
            self.loss = self.reconstruction_loss + self.vae.regularization_loss * regularizer_weight
            self.mu_var = tf.math.reduce_variance(self.vae.mu, axis=0)
            self.logsig = tf.reduce_mean(self.vae.logsig, axis=0)


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
        self.log_loss_rec = 0.
        self.log_loss_reg = 0.
        self.log_loss_muvar = np.array([0.] * 8)
        self.log_loss_logsig = np.array([0.] * 8)
        self.log_num = 0

    def network_update(self):
        self.log_loss_rec = 0.
        self.log_loss_reg = 0.
        self.log_loss_muvar = np.array([0.] * 8)
        self.log_loss_logsig = np.array([0.] * 8)
        self.log_num = 0
            
    def optimize_batch(self, input_state, input_route, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_route: input_route, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        _, l1, l2, l3, l4 = sess.run([self.train_action, self.reconstruction_loss, self.vae.regularization_loss,
                                      self.mu_var, self.logsig],input_list)

        self.log_loss_rec = l1
        self.log_loss_reg = l2
        self.log_loss_muvar = l3
        self.log_loss_logsig = l4
        self.log_num += 1

    def log_caption(self):
        return "\t"  + self.name + "_ReconLoss\t"  + self.name + "_RegLoss\t"  \
            + self.name + "_MuVar\t\t\t\t\t\t\t\t" + self.name + "_Logsig\t\t\t\t\t\t\t"

    def current_log(self):
        return "\t" + str(self.log_loss_rec / self.log_num) + "\t" + str(self.log_loss_reg / self.log_num) + "\t" \
            + "\t".join([str(self.log_loss_muvar[i] / self.log_num) for i in range(8)]) + "\t" \
            + "\t".join([str(self.log_loss_logsig[i] / self.log_num) for i in range(8)])

    def log_print(self):
        print ( self.name \
            + "\n\tReconLoss         : " + str(self.log_loss_rec / self.log_num) \
            + "\n\tRegLoss           : " + str(self.log_loss_reg / self.log_num) \
            + "\n\tMuVar             : " + " ".join([str(self.log_loss_muvar[i] / self.log_num)[:8] for i in range(8)]) \
            + "\n\tLogsig            : " + " ".join([str(self.log_loss_logsig[i] / self.log_num)[:8] for i in range(8)]))
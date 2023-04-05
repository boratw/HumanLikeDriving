import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Reverse
from network.vencoder import VEncoder
from network.onedcnn import OneDCnn



class TrajectoryEstimator():
    def __init__(self, name=None, reuse=False, global_learner_lr=0.0001, local_learner_lr = 0.00005, global_latent_len = 4, local_latent_len = 4, global_input_latent_length=100, traj_len = 5, 
            reconstruction_loss_magnifier=1., global_different_loss_magnifier = 1.0, global_regularizer_weight=0.001, local_regularizer_weight = 0.01):

        if name == None:
            self.name = "TrajectoryEstimator"
        else:
            self.name = "TrajectoryEstimator" + name
        self.global_latent_len = global_latent_len
        self.local_latent_len = local_latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, 4])
            self.layer_input_waypoints = tf.placeholder(tf.float32, [None, 3, 5])
            self.layer_input_othervcs = tf.placeholder(tf.float32, [None, 8, 5])
            self.layer_input_traj = tf.placeholder(tf.float32, [None, traj_len, 2])
            self.layer_input_global = tf.placeholder(tf.float32, [None, global_input_latent_length, local_latent_len])

            layer_input_waypoints_reshaped = tf.reshape(self.layer_input_waypoints, [-1, 15])
            layer_input_othervcs_reshaped = tf.reshape(self.layer_input_othervcs, [-1, 40])
            layer_input_global_reshaped = tf.reshape(self.layer_input_global, [-1, 1, global_input_latent_length, local_latent_len])

            self.layer_input = tf.concat([self.layer_input_state, layer_input_waypoints_reshaped, layer_input_othervcs_reshaped], axis=1)
            self.layer_traj = tf.reshape(self.layer_input_traj, [-1, traj_len * 2])

            self.local_vae = VAE(self.layer_traj.shape[1], local_latent_len, [128, 64, 128], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_traj,
                           additional_dim=self.layer_input.shape[1], additional_tensor=self.layer_input, name="LocalVAE" )

            self.global_encoder = OneDCnn(local_latent_len, global_latent_len, [32, 64, 96, 128], [128, 64, 32], hidden_nonlns = tf.nn.leaky_relu, 
                        input_tensor=layer_input_global_reshaped, name="GlobalEncoder" )

            self.local_reconstruction_loss = tf.reduce_mean((self.local_vae.layer_output - self.layer_traj) ** 2)
            self.local_loss = self.local_reconstruction_loss + self.local_vae.regularization_loss * local_regularizer_weight

            self.global_encoded_latent_size = tf.math.sqrt(tf.reduce_sum(self.global_encoder.layer_output ** 2, axis=1, keep_dims=True))
            self.global_encoded_latent = self.global_encoder.layer_output / tf.stop_gradient(self.global_encoded_latent_size + 1e-7)
            global_encoded_latents = tf.split(self.global_encoded_latent, 4, axis=0)
            global_similar_loss = []
            global_different_loss = []
            for i, latent1 in enumerate(global_encoded_latents):
                latent_mean = tf.reduce_mean(latent1, axis=0, keepdims=True)
                for j, latent2 in enumerate(global_encoded_latents):
                    if i == j:
                        global_similar_loss.append(tf.reduce_mean((latent_mean - latent2) ** 2))
                    else:
                        global_different_loss.append(tf.reduce_mean((latent_mean - latent2) ** 2))
            self.global_similar_loss = tf.reduce_mean(global_similar_loss)
            self.global_different_loss = tf.reduce_mean(global_different_loss)
            self.global_regularizer_loss = tf.reduce_mean(self.global_encoded_latent_size ** 2)

            self.global_loss = self.global_similar_loss - self.global_different_loss * global_different_loss_magnifier + self.global_regularizer_loss * global_regularizer_weight

            self.global_mu_var = tf.math.reduce_variance(self.global_encoded_latent, axis=0)

            self.local_mu_var = tf.math.reduce_variance(self.local_vae.mu, axis=0)
            self.local_logsig = tf.reduce_mean(self.local_vae.logsig, axis=0)

            self.global_optimizer = tf.train.AdamOptimizer(global_learner_lr)
            self.local_optimizer = tf.train.AdamOptimizer(local_learner_lr)
            self.global_train_action = self.global_optimizer.minimize(loss = tf.reduce_sum(self.global_loss), var_list=self.global_encoder.trainable_params)
            self.local_train_action = self.local_optimizer.minimize(loss = tf.reduce_sum(self.local_loss), var_list=self.local_vae.trainable_params)


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}

    def network_initialize(self):
        self.log_global_loss_sim = 0.
        self.log_global_loss_diff = 0.
        self.log_global_loss_regul = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_local_loss_rec = 0.
        self.log_local_loss_regul = 0.
        self.log_local_muvar = np.array([0.] * self.local_latent_len)
        self.log_local_logsig = np.array([0.] * self.local_latent_len)
        self.log_local_num = 0
        self.log_global_num = 0

    def network_update(self):
        self.log_global_loss_sim = 0.
        self.log_global_loss_diff = 0.
        self.log_global_loss_regul = 0.
        self.log_global_muvar = np.array([0.] * self.global_latent_len)
        self.log_global_num = 0
        self.log_local_loss_rec = 0.
        self.log_local_loss_regul = 0.
        self.log_local_muvar = np.array([0.] * self.local_latent_len)
        self.log_local_logsig = np.array([0.] * self.local_latent_len)
        self.log_local_num = 0
            
    def optimize_local(self, input_state, input_waypoints, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        _, l0, l1, l2, l3, l4, l5 = sess.run([self.local_train_action, self.local_loss, self.local_vae.mu,
                                    self.local_reconstruction_loss, self.local_vae.regularization_loss,
                                    self.local_mu_var, self.local_logsig ],input_list)
        
        self.log_local_loss_rec += l2
        self.log_local_loss_regul += l3
        self.log_local_muvar += l4
        self.log_local_logsig += l5
        self.log_local_num += 1

        return l1, l0
    
    def optimize_global(self, input_global):
        input_list = {self.layer_input_global : input_global}
        sess = tf.get_default_session()
        _, l0, l1, l2, l3, l4 = sess.run([self.global_train_action, self.global_loss, self.global_similar_loss, self.global_different_loss,
                                    self.global_regularizer_loss, self.global_mu_var ],input_list)
        self.log_global_loss_sim += l1
        self.log_global_loss_diff += l2
        self.log_global_loss_regul += l3
        self.log_global_muvar += l4

        self.log_global_num += 1
        return l0
    
    '''
    def get_global_latents(self, input_state, input_waypoints, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        l0 = sess.run([self.global_encoded_latent], input_list)
        return l0[0]
    
    def get_local_latents(self, input_state, input_waypoints, input_othervcs, input_traj, input_global_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj, }
        sess = tf.get_default_session()
        mu, var = sess.run([self.local_vae_global_target.mu, self.local_vae_global_target.sig], input_list)
        return mu, var

    def get_routes(self, input_state, input_waypoints, input_othervcs, input_global_latent, input_local_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_global_latent : input_global_latent,
                    self.local_vae_global_target.layer_latent : input_local_latent }
        sess = tf.get_default_session()
        res = sess.run(self.route_output, input_list)
        return res
    '''
    
    def log_caption(self):
        return "\t" + self.name + "_GlobalSimLoss\t"  + self.name + "_GlobalDiffLoss\t" + self.name + "_GlobalRegulLoss\t" \
            + self.name + "_GlobalLogsig\t" + "\t".join([ "" for _ in range(self.global_latent_len)]) \
            + self.name + "_LocalReconLoss\t"  + self.name + "_LocalRegulLoss\t" \
            + self.name + "_LocalMuVar\t" + "\t".join([ "" for _ in range(self.local_latent_len)]) \
            + self.name + "_LocalLogsig\t" + "\t".join([ "" for _ in range(self.local_latent_len)]) 

    def current_log(self):
        return "\t" + str(self.log_global_loss_sim / self.log_global_num) + "\t" + str(self.log_global_loss_diff / self.log_global_num) + "\t" \
            + str(self.log_global_loss_regul / self.log_global_num)  \
            + "\t".join([str(self.log_global_muvar[i] / self.log_global_num) for i in range(self.global_latent_len)]) + "\t" \
            + "\t" + str(self.log_local_loss_rec / self.log_local_num) + "\t" + str(self.log_local_loss_regul / self.log_local_num) + "\t" \
            + "\t".join([str(self.log_local_muvar[i] / self.log_local_num) for i in range(self.local_latent_len)]) + "\t" \
            + "\t".join([str(self.log_local_logsig[i] / self.log_local_num) for i in range(self.local_latent_len)])

    def log_print(self):
        print ( self.name \
            + "\n\tGlobalSimLoss          : " + str(self.log_global_loss_sim / self.log_global_num) \
            + "\n\tGlobalDiffLoss          : " + str(self.log_global_loss_diff / self.log_global_num) \
            + "\n\tGlobalRegulLoss        : " + str(self.log_global_loss_regul / self.log_global_num) \
            + "\n\tGlobalMuVar            : " + " ".join([str(self.log_global_muvar[i] / self.log_global_num)[:8] for i in range(self.global_latent_len)]) \
            + "\n\tLocalReconLoss         : " + str(self.log_local_loss_rec / self.log_local_num) \
            + "\n\tLocalRegulLoss         : " + str(self.log_local_loss_regul / self.log_local_num) \
            + "\n\tLocalMuVar             : " + " ".join([str(self.log_local_muvar[i] / self.log_local_num)[:8] for i in range(self.local_latent_len)]) \
            + "\n\tLocalLogsig            : " + " ".join([str(self.log_local_logsig[i] / self.log_local_num)[:8] for i in range(self.local_latent_len)]))
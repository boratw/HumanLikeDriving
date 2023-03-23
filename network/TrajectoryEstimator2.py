import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Reverse



class TrajectoryEstimator():
    def __init__(self, name=None, reuse=False, learner_lr=0.0001, global_latent_len = 4, local_latent_len = 4, traj_len = 5, different_loss_magnifier=0.1, reconstruction_loss_magnifier=1.,
            global_regularizer_weight=0.1, local_regularizer_weight = 0.01, regenerate_weight=0.1):

        if name == None:
            self.name = "TrajectoryEstimator"
        else:
            self.name = "TrajectoryEstimator" + name
        self.global_latent_len = global_latent_len
        self.local_latent_len = local_latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, 2])
            self.layer_input_waypoints = tf.placeholder(tf.float32, [None, 3, 5])
            self.layer_input_othervcs = tf.placeholder(tf.float32, [None, 8, 5])
            self.layer_input_traj = tf.placeholder(tf.float32, [None, traj_len, 2])
            self.layer_target_global_latent = tf.placeholder(tf.float32, [None, global_latent_len])

            layer_input_waypoints_reshaped = tf.reshape(self.layer_input_waypoints, [-1, 15])
            layer_input_othervcs_reshaped = tf.reshape(self.layer_input_othervcs, [-1, 40])

            self.layer_input = tf.concat([self.layer_input_state, layer_input_waypoints_reshaped, layer_input_othervcs_reshaped], axis=1)
            self.layer_output = tf.reshape(self.layer_input_traj, [-1, traj_len * 2])
            self.layer_global_input = tf.concat([self.layer_input, self.layer_output], axis=1)

            self.global_encoder = MLP(traj_len * 2 + 57, global_latent_len, [128, 64, 32], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_global_input, name="GlobalEncoder" )
            self.layer_local_input = tf.concat([tf.stop_gradient(self.global_encoder.layer_output), self.layer_input], axis=1)

            self.local_vae = VAE(traj_len * 2, local_latent_len, [128, 64, 128], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_output,
                           additional_dim=57 + global_latent_len, additional_tensor=self.layer_local_input, name="LocalVAE" )
            
            self.random_input_dist = tf.distributions.Normal(loc=tf.zeros_like(self.global_encoder.layer_output), scale=tf.ones_like(self.global_encoder.layer_output))
            self.random_global_latent = tf.stop_gradient(self.random_input_dist.sample())
            self.random_local_input = tf.concat([self.random_global_latent, self.layer_input], axis=1)
            self.local_vae_random = VAE(traj_len * 2, local_latent_len, [128, 64, 128], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.layer_output,
                           additional_dim=57 + global_latent_len, additional_tensor=self.random_local_input, name="LocalVAE", reuse=True)
            self.random_global_input = tf.concat([self.layer_input, self.local_vae_random.layer_output], axis=1)
            self.global_encoder_random = MLP(traj_len * 2 + 57, global_latent_len, [128, 64, 32], hidden_nonlns = tf.nn.leaky_relu, input_tensor=self.random_global_input, name="GlobalEncoder", reuse=True )

            self.global_similar_loss = tf.reduce_mean((self.global_encoder.layer_output - self.layer_target_global_latent) ** 2)
            self.global_different_loss = tf.reduce_mean(tf.math.reduce_variance(self.global_encoder.layer_output, axis=0))
            self.global_regularizer_loss = tf.reduce_mean(self.global_encoder.layer_output ** 2)
            self.global_loss = self.global_similar_loss  - self.global_different_loss * different_loss_magnifier +  self.global_regularizer_loss * global_regularizer_weight

            
            self.local_reconstruction_loss = tf.reduce_mean((self.local_vae.layer_output - self.layer_output) ** 2)
            self.local_regenerate_loss = tf.reduce_mean((self.global_encoder_random.layer_output - self.random_global_latent) ** 2)
            self.local_loss = self.local_reconstruction_loss * reconstruction_loss_magnifier + self.local_vae.regularization_loss * local_regularizer_weight + tf.reduce_mean(self.local_regenerate_loss)  * regenerate_weight

                
            self.local_mu_var = tf.math.reduce_variance(self.local_vae.mu, axis=0)
            self.local_logsig = tf.reduce_mean(self.local_vae.logsig, axis=0)

            self.route_output = tf.reshape(self.local_vae.latent_decoder.layer_output, ([-1, traj_len, 2]))

            self.global_optimizer = tf.train.AdamOptimizer(learner_lr)
            self.local_optimizer = tf.train.AdamOptimizer(learner_lr)
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
        self.log_global_num = 0
        self.log_local_loss_rec = 0.
        self.log_local_loss_regul = 0.
        self.log_local_loss_muvar = np.array([0.] * self.local_latent_len)
        self.log_local_loss_logsig = np.array([0.] * self.local_latent_len)
        self.log_local_loss_regen = np.array([0.] * self.global_latent_len)
        self.log_local_num = 0

    def network_update(self):
        self.log_global_loss_sim = 0.
        self.log_global_loss_diff = 0.
        self.log_global_loss_regul = 0.
        self.log_global_num = 0
        self.log_local_loss_rec = 0.
        self.log_local_loss_regul = 0.
        self.log_local_loss_muvar = np.array([0.] * self.local_latent_len)
        self.log_local_loss_logsig = np.array([0.] * self.local_latent_len)
        self.log_local_loss_regen = np.array([0.] * self.global_latent_len)
        self.log_local_num = 0
            
    def optimize_local(self, input_state, input_waypoints, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        _, l0, l1, l2, l3, l4, l5, l6 = sess.run([self.local_train_action, self.local_loss, self.global_encoder.layer_output,
                                    self.local_reconstruction_loss, self.local_vae.regularization_loss, self.local_regenerate_loss,
                                    self.local_mu_var, self.local_logsig ],input_list)
        
        self.log_local_loss_rec += l2
        self.log_local_loss_regul += l3
        self.log_local_loss_regen += l4
        self.log_local_loss_muvar += l5
        self.log_local_loss_logsig += l6
        self.log_local_num += 1
        return l1, l0
    
    def optimize_global(self, input_state, input_waypoints, input_othervcs, input_traj, input_target):
        input_list = {self.layer_input_state : input_state, self.layer_input_waypoints: input_waypoints, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj, self.layer_target_global_latent : input_target}
        sess = tf.get_default_session()
        _, l0, l1, l2, l3 = sess.run([self.global_train_action, self.global_loss, self.global_similar_loss,
                                    self.global_different_loss, self.global_regularizer_loss],input_list)
        self.log_global_loss_sim += l1
        self.log_global_loss_diff += l2
        self.log_global_loss_regul += l3
        self.log_global_num += 1

        return l0
    

    def log_caption(self):
        return "\t" + self.name + "_GlobalSimLoss\t" + self.name + "_GlobalDiffLoss\t" + self.name + "_GlobalRegulLoss\t" \
            + self.name + "_LocalReconLoss\t"  + self.name + "_LocalRegulLoss\t" \
            + self.name + "_LocalMuVar" + "\t".join([ "" for _ in range(self.local_latent_len)]) \
            + self.name + "_LocalLogsig" + "\t".join([ "" for _ in range(self.local_latent_len)]) \
            + self.name + "_LocalRegenLoss" + "\t".join([ "" for _ in range(self.global_latent_len)])

    def current_log(self):
        return "\t" + str(self.log_global_loss_sim / self.log_global_num) + "\t" + str(self.log_global_loss_diff / self.log_global_num)  \
            + "\t" + str(self.log_global_loss_regul / self.log_global_num) \
            + "\t" + str(self.log_local_loss_rec / self.log_local_num) + "\t" + str(self.log_local_loss_regul / self.log_local_num) + "\t" \
            + "\t".join([str(self.log_local_loss_muvar[i] / self.log_local_num) for i in range(self.local_latent_len)]) + "\t" \
            + "\t".join([str(self.log_local_loss_logsig[i] / self.log_local_num) for i in range(self.local_latent_len)])+ "\t" \
            + "\t".join([str(self.log_local_loss_regen[i] / self.log_local_num) for i in range(self.global_latent_len)])

    def log_print(self):
        print ( self.name \
            + "\n\tGlobalSimLoss          : " + str(self.log_global_loss_sim / self.log_global_num) \
            + "\n\tGlobalDiffLoss         : " + str(self.log_global_loss_diff / self.log_global_num) \
            + "\n\tGlobalRegulLoss        : " + str(self.log_global_loss_regul / self.log_global_num) \
            + "\n\tLocalReconLoss         : " + str(self.log_local_loss_rec / self.log_local_num) \
            + "\n\tLocalRegulLoss         : " + str(self.log_local_loss_regul / self.log_local_num) \
            + "\n\tLocalMuVar             : " + " ".join([str(self.log_local_loss_muvar[i] / self.log_local_num)[:8] for i in range(self.local_latent_len)]) \
            + "\n\tLocalLogsig            : " + " ".join([str(self.log_local_loss_logsig[i] / self.log_local_num)[:8] for i in range(self.local_latent_len)]) \
            + "\n\tLocalRegenLoss         : " + " ".join([str(self.log_local_loss_regen[i] / self.log_local_num)[:8] for i in range(self.global_latent_len)]))
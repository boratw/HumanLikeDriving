import tensorflow.compat.v1 as tf
import numpy as np
from network.mlp import MLP
from network.vae import VAE, VAE_Reverse



class TrajectoryEstimator():
    def __init__(self, name=None, reuse=False, learner_lr=0.0001, latent_len = 8, traj_len = 5, loss_magnifier=1., regularizer_weight=0.001, use_regen_loss=False, regenerate_weight=0.01):

        if name == None:
            self.name = "TrajectoryEstimator"
        else:
            self.name = "TrajectoryEstimator" + name
        self.use_regen_loss = use_regen_loss
        self.latent_len = latent_len
        
        with tf.variable_scope(self.name, reuse=reuse):

            self.layer_input_state = tf.placeholder(tf.float32, [None, 2])
            self.layer_input_route = tf.placeholder(tf.float32, [None, 3, 10])
            self.layer_input_othervcs = tf.placeholder(tf.float32, [None, 8, 5])
            self.layer_input_traj = tf.placeholder(tf.float32, [None, traj_len, 2])

            layer_input_route_reshaped = tf.reshape(self.layer_input_route, [-1, 30])

            layer_input_othervcs = tf.unstack(self.layer_input_othervcs, axis=1)
            othervcs_embeddings = []
            for i, layer in enumerate(layer_input_othervcs):
                outlayer = MLP(5, 8, [32], hidden_nonlns = tf.nn.tanh, input_tensor=layer, name="OtherVcsEmbedding" + str(i))
                othervcs_embeddings.append(outlayer.layer_output)
            self.othervcs_embedding = tf.reduce_sum(othervcs_embeddings, axis=0)

            self.layer_input = tf.concat([self.layer_input_state, layer_input_route_reshaped, self.othervcs_embedding], axis=1)
            self.layer_output = tf.reshape(self.layer_input_traj, [-1, traj_len * 2])

            self.vae = VAE(traj_len * 2, latent_len, [128, 64, 128], hidden_nonlns = tf.nn.tanh, input_tensor=self.layer_output,
                           additional_dim=40, additional_tensor=self.layer_input )
            
            if use_regen_loss:
                self.random_input_dist = tf.distributions.Normal(loc=tf.zeros_like(self.vae.mu), scale=tf.ones_like(self.vae.logsig))
                self.random_input = self.random_input_dist.sample()
                self.reverse_vae = VAE_Reverse(traj_len * 2, latent_len, [128, 64, 128], hidden_nonlns = tf.nn.tanh, input_tensor=self.random_input,
                            additional_dim=40, additional_tensor=self.layer_input, reuse=True )

            
            self.reconstruction_loss = tf.reduce_mean((self.vae.layer_output - self.layer_output) ** 2, axis=1)
            self.loss = tf.reduce_mean(self.reconstruction_loss) * loss_magnifier + self.vae.regularization_loss * regularizer_weight
            if use_regen_loss:
                self.regenerate_loss = tf.reduce_mean(self.reverse_vae.dist.log_prob(self.random_input), axis=0)
                self.loss -= tf.reduce_mean(self.regenerate_loss)  * regenerate_weight
            self.mu_var = tf.math.reduce_variance(self.vae.mu, axis=0)
            self.logsig = tf.reduce_mean(self.vae.logsig, axis=0)

            self.route_output = tf.reshape(self.vae.latent_decoder.layer_output, ([-1, traj_len, 2]))

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
        self.log_loss_regul = 0.
        self.log_loss_muvar = np.array([0.] * self.latent_len)
        self.log_loss_logsig = np.array([0.] * self.latent_len)
        self.log_loss_regen = np.array([0.] * self.latent_len)
        self.log_num = 0

    def network_update(self):
        self.log_loss_rec = 0.
        self.log_loss_regul = 0.
        self.log_loss_muvar = np.array([0.] * self.latent_len)
        self.log_loss_logsig = np.array([0.] * self.latent_len)
        self.log_loss_regen = np.array([0.] * self.latent_len)
        self.log_num = 0
            
    def optimize_batch(self, input_state, input_route, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_route: input_route, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        if self.use_regen_loss:
            _, l1, l2, l3, l4, l5 = sess.run([self.train_action, self.reconstruction_loss, self.vae.regularization_loss,
                                        self.mu_var, self.logsig, self.regenerate_loss],input_list)
            self.log_loss_regen += l5
        else:
            _, l1, l2, l3, l4 = sess.run([self.train_action, self.reconstruction_loss, self.vae.regularization_loss,
                                        self.mu_var, self.logsig],input_list)
        
        self.log_loss_rec += np.mean(l1)
        self.log_loss_regul += l2
        self.log_loss_muvar += l3
        self.log_loss_logsig += l4
        self.log_num += 1
        return l1
    
    def get_latents(self, input_state, input_route, input_othervcs, input_traj):
        input_list = {self.layer_input_state : input_state, self.layer_input_route: input_route, 
                      self.layer_input_othervcs : input_othervcs, self.layer_input_traj : input_traj}
        sess = tf.get_default_session()
        mu, var = sess.run([self.vae.mu, self.vae.sig], input_list)
        return mu, var
    
    def get_routes(self, input_state, input_route, input_othervcs, input_latent):
        input_list = {self.layer_input_state : input_state, self.layer_input_route: input_route, 
                      self.layer_input_othervcs : input_othervcs, self.vae.layer_latent : input_latent}
        sess = tf.get_default_session()
        res = sess.run(self.route_output, input_list)
        return res

    def log_caption(self):
        return "\t"  + self.name + "_ReconLoss\t"  + self.name + "_RegulLoss\t" \
            + self.name + "_MuVar" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_Logsig" + "\t".join([ "" for _ in range(self.latent_len)]) \
            + self.name + "_RegenLoss\t\t\t\t\t\t\t"

    def current_log(self):
        return "\t" + str(self.log_loss_rec / self.log_num) + "\t" + str(self.log_loss_regul / self.log_num) + "\t" \
            + "\t".join([str(self.log_loss_muvar[i] / self.log_num) for i in range(self.latent_len)]) + "\t" \
            + "\t".join([str(self.log_loss_logsig[i] / self.log_num) for i in range(self.latent_len)])+ "\t" \
            + "\t".join([str(self.log_loss_regen[i] / self.log_num) for i in range(self.latent_len)])

    def log_print(self):
        print ( self.name \
            + "\n\tReconLoss         : " + str(self.log_loss_rec / self.log_num) \
            + "\n\tRegulLoss         : " + str(self.log_loss_regul / self.log_num) \
            + "\n\tMuVar             : " + " ".join([str(self.log_loss_muvar[i] / self.log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tLogsig            : " + " ".join([str(self.log_loss_logsig[i] / self.log_num)[:8] for i in range(self.latent_len)]) \
            + "\n\tRegenLoss         : " + " ".join([str(self.log_loss_regen[i] / self.log_num)[:8] for i in range(self.latent_len)]))
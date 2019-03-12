import tensorflow as tf
import numpy as np

from utils_amazon import load_amazon, split_data, MLP, identity, turn_tfidf, turn_one_hot
from adaptation import *
from baseline import *


class CoTrainer(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd_extend/Cotrain_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.common_model_save_to = 'output/model/cmd_extend/Cotrain_common_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.target_model_save_to = 'output/model/cmd_extend/Cotrain_target_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.combined_model_save_to = 'output/model/cmd_extend/Cotrain_combined_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.n_input = 5000
        self.n_classes = 2
        self.d_hidden = 50
        self.tolerate_time = 20
        self.alpha = 1.
        self.belta = 1.
        self.gamma = 1.
        self.lamb = 1.
        self.beta = 1.
        self.recon = 1.
        self.n_hidden_c = 50
        self.n_hidden_s = 50
        self.n_hidden_t = 50

    def build_model(self):
        n_classes = self.n_classes

        def matchnorm(x1, x2):
            return tf.sqrt(tf.reduce_sum(tf.pow(x1 - x2, 2)))

        def scm(sx1, sx2, k):
            ss1 = tf.reduce_mean(tf.pow(sx1, k), 0)
            ss2 = tf.reduce_mean(tf.pow(sx2, k), 0)
            return matchnorm(ss1, ss2)

        def mmatch(x1, x2, n_moments):
            mx1 = tf.reduce_mean(x1, 0)
            mx2 = tf.reduce_mean(x2, 0)
            sx1 = x1 - mx1
            sx2 = x2 - mx2
            dm = matchnorm(mx1, mx2)
            scms = dm
            for i in range(n_moments - 1):
                scms += scm(sx1, sx2, i + 2)
            return scms

        # tf Graph input
        self.X_s_u = tf.placeholder("float", [None, self.n_input])  # source unlabeled data
        self.X_t_u = tf.placeholder("float", [None, self.n_input])  # target unlabeled data
        self.X_s = tf.placeholder("float", [None, self.n_input])    # source labeled data
        self.Y_s = tf.placeholder("float", [None, n_classes])
        self.X_t = tf.placeholder("float", [None, self.n_input])    # target labeled data
        self.Y_t = tf.placeholder("float", [None, n_classes])

        self.common_encode_mlp = MLP(name='common_encode_mlp', dims=[self.n_input, self.n_hidden_c],
                                     activations=[tf.nn.sigmoid])
        self.target_encode_mlp = MLP(name='target_encode_mlp', dims=[self.n_input, self.n_hidden_t],
                                     activations=[tf.nn.sigmoid])
        self.common_decode_mlp = MLP(name='common_decode_mlp',
                                     dims=[self.n_hidden_c, (self.n_hidden_c + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.target_decode_mlp = MLP(name='target_decode_mlp',
                                     dims=[self.n_hidden_t, (self.n_hidden_t + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.common_output_mlp = MLP(name='common_output_mlp', dims=[self.n_hidden_c, n_classes],
                                     activations=[identity])
        self.target_output_mlp = MLP(name='target_output_mlp', dims=[self.n_hidden_t, n_classes],
                                     activations=[identity])

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 3)
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)   # change to the common decode mlp
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))
        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.pred_s = pred_s
        self.prob_s = tf.nn.softmax(pred_s)
        self.C_loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s)))
        correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Get target classification loss
        encoding_t_t = self.target_encode_mlp.apply(self.X_t)
        pred_t = self.target_output_mlp.apply(encoding_t_t)
        self.pred_t = pred_t
        self.prob_t = tf.nn.softmax(pred_t)
        self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_t, labels=self.Y_t))
        correct_prediction = tf.equal(tf.argmax(pred_t, 1), tf.argmax(self.Y_t, 1))
        self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.common_output_mlp.parameters +
                      self.target_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        for tensor in self.target_output_mlp.parameters + self.target_encode_mlp.parameters:
            if tensor.name.find('W') != 0:
                l2_norm += 4 * tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.recon * self.R_loss +
                     self.alpha * self.C_loss +
                     self.belta * self.T_loss +
                     self.gamma * self.cmd_c_loss +
                     self.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))
        self.common_loss = (self.recon * self.R_loss +
                            self.alpha * self.C_loss +
                            self.gamma * self.cmd_c_loss +
                            self.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        self.common_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.common_loss, var_list=self.theta))
        self.target_loss = (self.recon * self.R_loss +
                            self.belta * self.T_loss +
                            self.gamma * self.cmd_c_loss +
                            self.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        self.target_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.target_loss, var_list=self.theta))
        self.combined_loss = (self.C_loss + 0.0001 * l2_norm)
        self.combined_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.combined_loss, var_list=self.theta))

    def train_common_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, c_loss, cmd_c_loss, cmd_t_loss, r_loss, accuracy) = self.sess.run(
                    [self.common_solver, self.C_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss,
                     self.accuracy_s],
                    feed_dict={self.X_s: x_t,
                               self.Y_s: y_t,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('c_loss:{0}'.format(c_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
                print('r_loss:{0}'.format(r_loss))
                print('accuracy_s:{0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = self.accuracy_s.eval({self.X_s: x_valid,
                                                         self.Y_s: y_valid},
                                                        session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.common_model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.restore(self.sess, self.common_model_save_to)
            test_accuracy = self.accuracy_s.eval({self.X_s: x_test,
                                                   self.Y_s: y_test},
                                                  session=self.sess)
            print('test_accuracy:{0}'.format(test_accuracy))
            return best_result

    def train_target_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, t_loss, cmd_c_loss, cmd_t_loss,
                 r_loss, accuracy) = self.sess.run(
                    [self.target_solver, self.T_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss, self.accuracy_t],
                    feed_dict={self.X_t: x_t,
                               self.Y_t: y_t,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                    }
                )
                print('t_loss:{0}'.format(t_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
                print('r_loss:{0}'.format(r_loss))
                print('accuracy_t:{0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = self.accuracy_t.eval({self.X_t: x_valid,
                                                           self.Y_t: y_valid},
                                                          session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.target_model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.restore(self.sess, self.target_model_save_to)
            test_accuracy = self.accuracy_t.eval({self.X_t: x_test,
                                                  self.Y_t: y_test},
                                                 session=self.sess)
            print('test_accuracy:{0}'.format(test_accuracy))
            return best_result

    def train_combined_model(self, x_t, y_t, x_valid, y_valid, x_test, y_test):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            common_valid_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_valid})
            common_test_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_test})
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            target_valid_probs, = self.sess.run([self.prob_t], feed_dict={self.X_t: x_valid})
            target_test_probs, = self.sess.run([self.prob_t], feed_dict={self.X_t: x_test})
            best_result = 0.
            best_beta = 0.
            for beta in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
                valid_probs = common_valid_probs + beta * target_valid_probs
                valid_accuracy = np.equal(valid_probs.argmax(axis=1), y_valid.argmax(axis=1)).mean()
                if valid_accuracy > best_result:
                    best_result = valid_accuracy
                    best_beta = beta
            valid_accuracy = best_result
            test_probs = common_test_probs + best_beta * target_test_probs
            test_accuracy = np.equal(test_probs.argmax(axis=1), y_test.argmax(axis=1)).mean()
            print('valid accuracy:', valid_accuracy)
            print("test accuracy:", test_accuracy)
            return valid_accuracy, test_accuracy

    def train(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        U = np.copy(x_t_u)
        select_num = 5
        best_result = 0.
        final_test_accuracy = 0.
        wait_times = 0.
        while len(U) > 0:
            print('Train common model...')
            self.train_common_model(x_s_u, x_t_u, np.concatenate([x_s, x_t]), np.concatenate([y_s, y_t]),
                                    x_valid, y_valid, x_test, y_test)
            print('Train target model...')
            self.train_target_model(x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test)
            # Select U
            probs = [self.get_common_prediction(U), self.get_target_prediction(U)]
            x_hat, y_hat, U = self.select_sample(U, probs, select_num=select_num)
            x_t = np.concatenate([x_t, x_hat], axis=0)
            y_t = np.concatenate([y_t, y_hat], axis=0)
            print('Train combined model...')
            valid_accuracy, test_accuracy = self.train_combined_model(x_t, y_t, x_valid, y_valid, x_test, y_test)
            if valid_accuracy > best_result:
                best_result = valid_accuracy
                final_test_accuracy = test_accuracy
                wait_times = 0
            else:
                wait_times += 1
            if wait_times >= self.tolerate_time:
                print('best_result:{0}'.format(best_result))
                break
        print('Test accuracy:{0}'.format(final_test_accuracy))

    def select_sample(self, U, probs, select_num):
        neg_idxes = set()
        pos_idxes = set()
        left_idxes = set(range(len(U)))
        for prob in probs:
            idxes = np.argsort(prob[:, 0])
            end_idx = min(select_num, (prob[:, 0][idxes[:select_num]] < 0.5).sum())
            begin_idx = min(select_num, (prob[:, 0][idxes[-select_num:]] > 0.5).sum())
            idx = min(begin_idx, end_idx)
            if idx == 0:
                idx = 1
            begin_idx = idx
            end_idx = idx
            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            print('pos num:', len(pos_idxes))
            print('neg num:', len(neg_idxes))
            left_idxes = left_idxes.intersection(idxes[end_idx:-begin_idx])

        pos_idxes = np.array(list(pos_idxes))
        neg_idxes = np.array(list(neg_idxes))
        left_idxes = np.array(list(left_idxes))
        x_n = U[neg_idxes]
        x_p = U[pos_idxes]
        y_n = np.zeros(shape=(len(x_n), 2), dtype='float32')
        y_n[:, 1] = 1.
        y_p = np.zeros(shape=(len(x_p), 2), dtype='float32')
        y_p[:, 0] = 1.
        U = U[left_idxes]
        x = np.concatenate([x_n, x_p], axis=0)
        y = np.concatenate([y_n, y_p], axis=0)
        x, y = shuffle([x, y])
        print('unlabelled num:', len(U))
        print(len(left_idxes))
        return x, y, U

    def get_common_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: X})
        return probs

    def get_target_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            probs, = self.sess.run([self.prob_t], feed_dict={self.X_t: X})
        return probs

    def analysis(self, x_t_train, y_t_train):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, 'output/model/cmd/combined.pkl')
            H_t, = self.sess.run([self.encoding_t_t],
                                 feed_dict={self.X_t: x_t_train})
        pca = PCA(n_components=2)
        H_t_hat = pca.fit_transform(H_t)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==0.], H_t_hat[:, 1][y_t_train[:, 0]==0.], color='r', alpha=.4, s=1)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==1.], H_t_hat[:, 1][y_t_train[:, 0]==1.], color='b', alpha=.4, s=1)
        plt.savefig("h_t.pdf", dpi=72)


class CoCMD(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.classifier = CombinedClassifier()
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd_extend/Cotrain_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.common_model_save_to = 'output/model/cmd_extend/Cotrain_common_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.target_model_save_to = 'output/model/cmd_extend/Cotrain_target_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.combined_model_save_to = 'output/model/cmd_extend/Cotrain_combined_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.n_input = 5000
        self.n_classes = 2
        self.d_hidden = 50
        self.tolerate_time = 20
        self.alpha = 1.
        self.belta = 1.
        self.gamma = 1.
        self.lamb = 1.
        self.beta = 1.
        self.recon = 1.
        self.n_hidden_c = 50
        self.n_hidden_s = 50
        self.n_hidden_t = 50

    def build_model(self):
        n_classes = self.n_classes

        def matchnorm(x1, x2):
            return tf.sqrt(tf.reduce_sum(tf.pow(x1 - x2, 2)))

        def scm(sx1, sx2, k):
            ss1 = tf.reduce_mean(tf.pow(sx1, k), 0)
            ss2 = tf.reduce_mean(tf.pow(sx2, k), 0)
            return matchnorm(ss1, ss2)

        def mmatch(x1, x2, n_moments):
            mx1 = tf.reduce_mean(x1, 0)
            mx2 = tf.reduce_mean(x2, 0)
            sx1 = x1 - mx1
            sx2 = x2 - mx2
            dm = matchnorm(mx1, mx2)
            scms = dm
            for i in range(n_moments - 1):
                scms += scm(sx1, sx2, i + 2)
            return scms

        # tf Graph input
        self.X_s_u = tf.placeholder("float", [None, self.n_input])  # source unlabeled data
        self.X_t_u = tf.placeholder("float", [None, self.n_input])  # target unlabeled data
        self.X_s = tf.placeholder("float", [None, self.n_input])    # source labeled data
        self.Y_s = tf.placeholder("float", [None, n_classes])
        self.X_t = tf.placeholder("float", [None, self.n_input])    # target labeled data
        self.Y_t = tf.placeholder("float", [None, n_classes])

        self.common_encode_mlp = MLP(name='common_encode_mlp', dims=[self.n_input, self.n_hidden_c],
                                     activations=[tf.nn.sigmoid])
        self.target_encode_mlp = MLP(name='target_encode_mlp', dims=[self.n_input, self.n_hidden_t],
                                     activations=[tf.nn.sigmoid])
        self.common_decode_mlp = MLP(name='common_decode_mlp',
                                     dims=[self.n_hidden_c, (self.n_hidden_c + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.target_decode_mlp = MLP(name='target_decode_mlp',
                                     dims=[self.n_hidden_t, (self.n_hidden_t + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.common_output_mlp = MLP(name='common_output_mlp', dims=[self.n_hidden_c, n_classes],
                                     activations=[identity])
        self.target_output_mlp = MLP(name='target_output_mlp', dims=[self.n_hidden_t, n_classes],
                                     activations=[identity])

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 3)
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)   # change to the common decode mlp
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))
        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.pred_s = pred_s
        self.prob_s = tf.nn.softmax(pred_s)
        self.C_loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s)))
        correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Get target classification loss
        encoding_t_t = self.target_encode_mlp.apply(self.X_t)
        pred_t = self.target_output_mlp.apply(encoding_t_t)
        self.pred_t = pred_t
        self.prob_t = tf.nn.softmax(pred_t)
        self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_t, labels=self.Y_t))
        correct_prediction = tf.equal(tf.argmax(pred_t, 1), tf.argmax(self.Y_t, 1))
        self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.common_output_mlp.parameters +
                      self.target_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        for tensor in self.target_output_mlp.parameters + self.target_encode_mlp.parameters:
            if tensor.name.find('W') != 0:
                l2_norm += 4 * tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.recon * self.R_loss +
                     self.alpha * self.C_loss +
                     self.belta * self.T_loss +
                     self.gamma * self.cmd_c_loss +
                     self.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))

    def train_model(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, c_loss, t_loss, cmd_c_loss, cmd_t_loss,
                 r_loss, accuracy_s, accuracy_t) = self.sess.run(
                    [self.solver, self.C_loss, self.T_loss, self.cmd_c_loss, -self.cmd_t_loss,
                     self.R_loss, self.accuracy_s, self.accuracy_t],
                    feed_dict={self.X_t: x_t,
                               self.Y_t: y_t,
                               self.X_s: x_s,
                               self.Y_s: y_s,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('t_loss:{0}'.format(t_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
                print('r_loss:{0}'.format(r_loss))
                print('accuracy_s:{0}'.format(accuracy_s))
                print('accuracy_t:{0}'.format(accuracy_t))
                if accuracy_t > 0.7:
                    common_valid_preds, target_valid_preds = self.sess.run([self.prob_s, self.prob_t],
                                                        feed_dict={self.X_s: x_valid, self.X_t: x_valid})
                    valid_preds = common_valid_preds + target_valid_preds
                    valid_accuracy = np.equal(np.argmax(valid_preds, axis=1), np.argmax(y_valid, axis=1)).mean()
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.restore(self.sess, self.model_save_to)
            common_test_preds, target_test_preds = self.sess.run([self.prob_s, self.prob_t],
                                                                   feed_dict={self.X_s: x_test, self.X_t: x_test})
            test_preds = common_test_preds + target_test_preds
            test_accuracy = np.equal(np.argmax(test_preds, axis=1), np.argmax(y_test, axis=1)).mean()
            print('test_accuracy:{0}'.format(test_accuracy))
            return best_result, test_accuracy

    def train(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        U = np.copy(x_t_u)
        select_num = 5
        best_result = 0.
        final_test_accuracy = 0.
        wait_times = 0.
        while len(U) > 0:
            print('Train model...')
            valid_accuracy, test_accuracy = self.train_model(x_s_u, x_t_u,
                                                             np.concatenate([x_s, x_t]), np.concatenate([y_s, y_t]),
                                                             x_t, y_t, x_valid, y_valid, x_test, y_test)
            # Select unlabeled data
            probs = self.get_prediction(U)
            x_hat, y_hat, U = self.select_sample(U, probs, select_num=select_num)
            x_t = np.concatenate([x_t, x_hat], axis=0)
            y_t = np.concatenate([y_t, y_hat], axis=0)
            print('Train combined model...')
            if valid_accuracy > best_result:
                best_result = valid_accuracy
                final_test_accuracy = test_accuracy
                wait_times = 0
            else:
                wait_times += 1
            if wait_times >= self.tolerate_time:
                print('best_result:{0}'.format(best_result))
                break
        print('Test accuracy:{0}'.format(final_test_accuracy))

    def select_sample(self, U, probs, select_num):
        neg_idxes = set()
        pos_idxes = set()
        left_idxes = set(range(len(U)))
        for prob in probs:
            idxes = np.argsort(prob[:, 0])
            end_idx = min(select_num, (prob[:, 0][idxes[:select_num]] < 0.5).sum())
            begin_idx = min(select_num, (prob[:, 0][idxes[-select_num:]] > 0.5).sum())
            idx = min(begin_idx, end_idx)
            if idx == 0:
                idx = 1
            begin_idx = idx
            end_idx = idx
            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            print('pos num:', len(pos_idxes))
            print('neg num:', len(neg_idxes))
            left_idxes = left_idxes.intersection(idxes[end_idx:-begin_idx])

        pos_idxes = np.array(list(pos_idxes))
        neg_idxes = np.array(list(neg_idxes))
        left_idxes = np.array(list(left_idxes))
        x_n = U[neg_idxes]
        x_p = U[pos_idxes]
        y_n = np.zeros(shape=(len(x_n), 2), dtype='float32')
        y_n[:, 1] = 1.
        print(probs[0][neg_idxes])
        print(y_n)
        # raw_input('Input any character!')
        y_p = np.zeros(shape=(len(x_p), 2), dtype='float32')
        y_p[:, 0] = 1.
        U = U[left_idxes]
        x = np.concatenate([x_n, x_p], axis=0)
        y = np.concatenate([y_n, y_p], axis=0)
        print('unlabelled num:', len(U))
        print(len(left_idxes))
        return x, y, U

    def get_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.model_save_to)
            prob_s, prob_t = self.sess.run([self.prob_s, self.prob_t], feed_dict={self.X_s: X, self.X_t: X})
        return [prob_s, prob_t]

    def analysis(self, x_t_train, y_t_train):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, 'output/model/cmd/combined.pkl')
            H_t, = self.sess.run([self.encoding_t_t],
                                 feed_dict={self.X_t: x_t_train})
        pca = PCA(n_components=2)
        H_t_hat = pca.fit_transform(H_t)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==0.], H_t_hat[:, 1][y_t_train[:, 0]==0.], color='r', alpha=.4, s=1)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==1.], H_t_hat[:, 1][y_t_train[:, 0]==1.], color='b', alpha=.4, s=1)
        plt.savefig("h_t.pdf", dpi=72)


class CoDSN(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd_extend/Cotrain_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.common_model_save_to = 'output/model/cmd_extend/Cotrain_common_{0}_to_{1}.pkl'.format(source_domain,
                                                                                                   target_domain)
        self.common_tune_model_save_to = 'output/model/cmd_extend/Cotrain_common_tune_{0}_to_{1}.pkl'.format(
            source_domain, target_domain)
        self.target_model_save_to = 'output/model/cmd_extend/Cotrain_target_{0}_to_{1}.pkl'.format(source_domain,
                                                                                                   target_domain)
        self.combined_model_save_to = 'output/model/cmd_extend/Cotrain_combined_{0}_to_{1}.pkl'.format(source_domain,
                                                                                                       target_domain)
        self.n_input = 5000
        self.n_classes = 2
        self.d_hidden = 50
        self.tolerate_time = 20
        self.alpha = 1.
        self.belta = 1.
        self.gamma = 1.
        self.lamb = 1.
        self.beta = 1.
        self.n_hidden_c = 50
        self.n_hidden_s = 50
        self.n_hidden_t = 50

    def build_model(self):
        n_classes = self.n_classes

        def matchnorm(x1, x2):
            return tf.sqrt(tf.reduce_sum(tf.pow(x1 - x2, 2)))

        def scm(sx1, sx2, k):
            ss1 = tf.reduce_mean(tf.pow(sx1, k), 0)
            ss2 = tf.reduce_mean(tf.pow(sx2, k), 0)
            return matchnorm(ss1, ss2)

        def mmatch(x1, x2, n_moments):
            mx1 = tf.reduce_mean(x1, 0)
            mx2 = tf.reduce_mean(x2, 0)
            sx1 = x1 - mx1
            sx2 = x2 - mx2
            dm = matchnorm(mx1, mx2)
            scms = dm
            for i in range(n_moments - 1):
                scms += scm(sx1, sx2, i + 2)
            return scms

        # tf Graph input
        self.X_s_u = tf.placeholder("float", [None, self.n_input])  # source unlabeled data
        self.X_t_u = tf.placeholder("float", [None, self.n_input])  # target unlabeled data
        self.X_s = tf.placeholder("float", [None, self.n_input])  # source labeled data
        self.Y_s = tf.placeholder("float", [None, n_classes])
        self.X_t = tf.placeholder("float", [None, self.n_input])  # target labeled data
        self.Y_t = tf.placeholder("float", [None, n_classes])

        self.common_encode_mlp = MLP(name='common_encode_mlp', dims=[self.n_input, self.n_hidden_c],
                                     activations=[tf.nn.sigmoid])
        self.target_encode_mlp = MLP(name='target_encode_mlp', dims=[self.n_input, self.n_hidden_t],
                                     activations=[tf.nn.sigmoid])
        self.common_decode_mlp = MLP(name='common_decode_mlp',
                                     dims=[self.n_hidden_c, (self.n_hidden_c + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.target_decode_mlp = MLP(name='target_decode_mlp',
                                     dims=[self.n_hidden_t, (self.n_hidden_t + self.n_input) / 2, self.n_input],
                                     activations=[tf.nn.tanh, tf.nn.relu])
        self.common_output_mlp = MLP(name='common_output_mlp', dims=[self.n_hidden_c, n_classes],
                                     activations=[identity])
        self.target_output_mlp = MLP(name='target_output_mlp', dims=[self.n_hidden_t, n_classes],
                                     activations=[identity])

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 5)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 5)
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))
        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.pred_s = pred_s
        self.prob_s = tf.nn.softmax(pred_s)
        self.C_loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s)))
        correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Get target classification loss
        encoding_t_t = self.target_encode_mlp.apply(self.X_t)
        pred_t = self.target_output_mlp.apply(encoding_t_t)
        self.pred_t = pred_t
        self.prob_t = tf.nn.softmax(pred_t)
        self.T_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_t, labels=self.Y_t))
        correct_prediction = tf.equal(tf.argmax(pred_t, 1), tf.argmax(self.Y_t, 1))
        self.accuracy_t = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.common_output_mlp.parameters +
                      self.target_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        for tensor in self.target_output_mlp.parameters:
            if tensor.name.find('W') != 0:
                l2_norm += 4 * tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.R_loss +
                     self.alpha * self.C_loss +
                     self.belta * self.T_loss +
                     self.gamma * self.cmd_c_loss +
                     self.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))
        self.common_loss = (self.R_loss +
                            self.alpha * self.C_loss +
                            self.gamma * self.cmd_c_loss +
                            self.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        self.common_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.common_loss, var_list=self.theta))
        self.target_loss = (self.R_loss +
                            self.belta * self.T_loss +
                            self.gamma * self.cmd_c_loss +
                            self.lamb * self.cmd_t_loss +
                            0.0001 * l2_norm)
        self.target_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.target_loss, var_list=self.theta))
        self.combined_loss = (self.C_loss + 0.0001 * l2_norm)
        self.combined_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                                .minimize(self.combined_loss, var_list=self.theta))

    def train_common_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, c_loss, cmd_c_loss, cmd_t_loss, r_loss, accuracy) = self.sess.run(
                    [self.common_solver, self.C_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss,
                     self.accuracy_s],
                    feed_dict={self.X_s: x_t,
                               self.Y_s: y_t,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('c_loss:{0}'.format(c_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
                print('r_loss:{0}'.format(r_loss))
                print('accuracy_s:{0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = self.accuracy_s.eval({self.X_s: x_valid,
                                                           self.Y_s: y_valid},
                                                          session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.common_model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.restore(self.sess, self.common_model_save_to)
            test_accuracy = self.accuracy_s.eval({self.X_s: x_test,
                                                  self.Y_s: y_test},
                                                 session=self.sess)
            print('test_accuracy:{0}'.format(test_accuracy))
            return best_result

    def train_target_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, t_loss, cmd_c_loss, cmd_t_loss,
                 r_loss, accuracy) = self.sess.run(
                    [self.target_solver, self.T_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss, self.accuracy_t],
                    feed_dict={self.X_t: x_t,
                               self.Y_t: y_t,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('t_loss:{0}'.format(t_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
                print('r_loss:{0}'.format(r_loss))
                print('accuracy_t:{0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy = self.accuracy_t.eval({self.X_t: x_valid,
                                                           self.Y_t: y_valid},
                                                          session=self.sess)
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.target_model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.restore(self.sess, self.target_model_save_to)
            test_accuracy = self.accuracy_t.eval({self.X_t: x_test,
                                                  self.Y_t: y_test},
                                                 session=self.sess)
            print('test_accuracy:{0}'.format(test_accuracy))
            return best_result

    def train_combined_model(self, x_t, y_t, x_valid, y_valid, x_test, y_test):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            common_valid_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_valid})
            common_test_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_test})
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            target_valid_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_valid})
            target_test_probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: x_test})
            best_result = 0.
            best_beta = 0.
            for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
                valid_probs = common_valid_probs + beta * target_valid_probs
                valid_accuracy = np.equal(valid_probs.argmax(axis=1), y_valid.argmax(axis=1)).mean()
                if valid_accuracy > best_result:
                    best_result = valid_accuracy
                    best_beta = beta
            valid_accuracy = best_result
            test_probs = common_test_probs + best_beta * target_test_probs
            test_accuracy = np.equal(test_probs.argmax(axis=1), y_test.argmax(axis=1)).mean()
            print('valid accuracy:', valid_accuracy)
            print("test accuracy:", test_accuracy)
            return valid_accuracy, test_accuracy

    def train(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        U = np.copy(x_t_u)
        select_num = 5
        best_result = 0.
        final_test_accuracy = 0.
        wait_times = 0.
        while len(U) > 0:
            print('Train common model...')
            self.train_common_model(x_s_u, x_t_u, np.concatenate([x_s, x_t]), np.concatenate([y_s, y_t]),
                                    x_valid, y_valid, x_test, y_test)
            # input = raw_input('Input andy character!')
            print('Train target model...')
            self.train_target_model(x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test)
            # input = raw_input('Input andy character!')
            # Select U from common view
            probs = [self.get_common_prediction(U), self.get_target_prediction(U)]
            x_hat, y_hat, U = self.select_sample(U, probs, select_num=select_num)
            x_t = np.concatenate([x_t, x_hat], axis=0)
            y_t = np.concatenate([y_t, y_hat], axis=0)
            print('Train combined model...')
            valid_accuracy, test_accuracy = self.train_combined_model(x_t, y_t, x_valid, y_valid, x_test, y_test)
            # input = raw_input('Input andy character!')
            if valid_accuracy > best_result:
                best_result = valid_accuracy
                final_test_accuracy = test_accuracy
                wait_times = 0
            else:
                wait_times += 1
            if wait_times >= self.tolerate_time:
                print('best_result:{0}'.format(best_result))
                break
        print('Test accuracy:{0}'.format(final_test_accuracy))

    def select_sample(self, U, probs, select_num):
        neg_idxes = set()
        pos_idxes = set()
        left_idxes = set(range(len(U)))
        for prob in probs:
            idxes = np.argsort(prob[:, 0])
            end_idx = min(select_num, (prob[:, 0][idxes[:select_num]] < 0.5).sum())
            begin_idx = min(select_num, (prob[:, 0][idxes[-select_num:]] > 0.5).sum())
            idx = min(begin_idx, end_idx)
            if idx == 0:
                idx = 1
            begin_idx = idx
            end_idx = idx
            neg_idxes.update(idxes[:end_idx])
            pos_idxes.update(idxes[-begin_idx:])
            print('pos num:', len(pos_idxes))
            print('neg num:', len(neg_idxes))
            left_idxes = left_idxes.intersection(idxes[end_idx:-begin_idx])

        pos_idxes = np.array(list(pos_idxes))
        neg_idxes = np.array(list(neg_idxes))
        left_idxes = np.array(list(left_idxes))
        x_n = U[neg_idxes]
        x_p = U[pos_idxes]
        y_n = np.zeros(shape=(len(x_n), 2), dtype='float32')
        y_n[:, 1] = 1.
        y_p = np.zeros(shape=(len(x_p), 2), dtype='float32')
        y_p[:, 0] = 1.
        U = U[left_idxes]
        x = np.concatenate([x_n, x_p], axis=0)
        y = np.concatenate([y_n, y_p], axis=0)
        x, y = shuffle([x, y])
        print('unlabelled num:', len(U))
        print(len(left_idxes))
        return x, y, U

    def get_common_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            probs, = self.sess.run([self.prob_s], feed_dict={self.X_s: X})
        return probs

    def get_target_prediction(self, X):
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            probs, = self.sess.run([self.prob_t], feed_dict={self.X_t: X})
        return probs

    def analysis(self, x_t_train, y_t_train):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, 'output/model/cmd/combined.pkl')
            H_t, = self.sess.run([self.encoding_t_t],
                                 feed_dict={self.X_t: x_t_train})
        pca = PCA(n_components=2)
        H_t_hat = pca.fit_transform(H_t)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0] == 0.], H_t_hat[:, 1][y_t_train[:, 0] == 0.], color='r', alpha=.4,
                    s=1)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0] == 1.], H_t_hat[:, 1][y_t_train[:, 0] == 1.], color='b', alpha=.4,
                    s=1)
        plt.savefig("h_t.pdf", dpi=72)


if __name__ == '__main__':
    data_load_from = 'data/cmd/amazon.mat'
    import sys
    source_domain = int(sys.argv[1])
    target_domain = int(sys.argv[2])
    x, y, offset = load_amazon(5000, data_load_from)
    x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(source_domain,
                                                                                    target_domain,
                                                                                    x, y, offset, 2000)
    x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_tr, x_t_tst], axis=0))
    x_s = x[:len(x_s_tr) + len(x_s_tst)]
    x_t = x[len(x_s):]

    x_s_tr = np.copy(x_s[:len(x_s_tr)])
    x_s_tst = np.copy(x_s[len(x_s_tr):])

    x_t_tr = np.copy(x_t[:len(x_t_tr)])
    x_t_tst = np.copy(x_t[len(x_t_tr):])

    x_t_tune = np.copy(x_t_tst[:5])
    y_t_tune = np.copy(y_t_tst[:5])
    x_t_tst = x_t_tst[50:]
    y_t_tst = y_t_tst[50:]

    x_t_valid = x_t_tst[:500]
    y_t_valid = y_t_tst[:500]
    x_t_tst = x_t_tst[500:]
    y_t_tst = y_t_tst[500:]

    trainer = CoTrainer(source_domain, target_domain)
    trainer.recon = 1.
    results = []
    results.append(trainer.train(x_s_tr, x_t_tr, np.copy(x_s_tr), y_s_tr, x_t_tune, y_t_tune, x_t_valid, y_t_valid, x_t_tst, y_t_tst))
    print('Reslut from {0} domain to {1} doamin:{2}'.format(source_domain, target_domain, mean_results[-1]))
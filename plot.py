import tensorflow as tf
import numpy as np
from utils_amazon import load_amazon, split_data, xavier_init, MLP, identity, turn_tfidf, turn_one_hot

from numpy.linalg import matrix_rank
from numpy.linalg import svd
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def shuffle(arrays):
    """
    shuffle data (used by split)
    """
    index_shuf = np.arange(arrays[0].shape[0])
    np.random.shuffle(index_shuf)
    return [array[index_shuf] for array in arrays]


def plot_dist(x_s, x_t, save_to):
    pca = PCA(n_components=2)
    x = np.concatenate([x_s, x_t])
    pca.fit(x)
    x_s_hat = pca.transform(x_s)
    x_t_hat = pca.transform(x_t)
    plt.scatter(x_s_hat[:, 0], x_s_hat[:, 1], color='r', alpha=.4, s=1)
    plt.scatter(x_t_hat[:, 0], x_t_hat[:, 1], color='b', alpha=.4, s=1)
    plt.savefig(save_to, dpi=72)
    plt.close()


class SourceOnly(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd/SourceOnly_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.n_input = 5000
        self.n_hidden_1 = 50
        self.n_classes = 2
        self.d_hidden = 50
        self.model_built = False
        self.tolerate_time = 20
        self.alpha = 1.
        self.sess = None

    def build_model(self):
        # tf.reset_default_graph()
        self.model_built = True
        n_hidden_1 = self.n_hidden_1
        n_classes = self.n_classes

        def encoding(x, weights, biases):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            return layer_1

        def predict(x, weights, biases):
            out_layer = tf.matmul(x, weights['out']) + biases['out']
            return out_layer

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(xavier_init([self.n_input, n_hidden_1])),
            'out': tf.Variable(xavier_init([n_hidden_1, n_classes])),
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros(shape=[n_hidden_1])),
            'out': tf.Variable(tf.zeros(shape=[n_classes])),
        }

        self.theta = list(self.weights.values()) + list(self.biases.values())

        self.encoding = encoding(self.X, self.weights, self.biases)
        self.pred = predict(self.encoding, self.weights, self.biases)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred, 1),
                                                        tf.argmax(self.Y, 1)), 'float'))
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,
                                                                             labels=self.Y))
        l2_norm = 0.
        for tensor in list(self.weights.values()):
            l2_norm += tf.reduce_sum(tf.abs(tensor))
        self.loss = self.C_loss + 0.0001 * l2_norm
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))

    def train(self, x_train, y_train, x_valid, y_valid, x_test, y_test, x_s_tst, x_t_tst):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.build_model()
            batch_size = self.batch_size
            # Initialize model
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            wait_times = 0
            best_result = 0.
            while True:
                _, loss_curr, accuracy = self.sess.run(
                [self.solver, self.loss, self.accuracy],
                feed_dict={self.X: x_train,
                           self.Y: y_train}
                )
                valid_accuracy = self.accuracy.eval({self.X: x_valid, self.Y: y_valid}, session=self.sess)
                if valid_accuracy > best_result:
                    best_result = valid_accuracy
                    wait_times = 0
                    print('save model...')
                    saver.save(self.sess, self.model_save_to)
                    # print('done!')
                else:
                    wait_times += 1
                if wait_times >= self.tolerate_time:
                    break
            saver.restore(self.sess, self.model_save_to)
            encoding_s, = self.sess.run(
                [self.encoding],
                feed_dict={self.X: x_s_tst,})
            encoding_t, = self.sess.run(
                [self.encoding],
                feed_dict={self.X: x_t_tst,})
            plot_dist(encoding_s, encoding_t, 'source_{0}_{1}.pdf'.format(self.source_domain, self.target_domain))


class SCMD(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd_extend/SCMD_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.n_input = 5000
        self.n_classes = 2
        self.model_built = False
        self.tolerate_time = 20
        self.alpha = 1.
        self.belta = 1.
        self.gamma = 2.
        self.n_hidden_c = 50
        self.n_hidden_s = 50
        self.n_hidden_t = 10

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

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))
        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s))
        correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.common_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.R_loss +
                     self.alpha * self.C_loss +
                     self.gamma * self.cmd_c_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))

    def train(self, x_s_u, x_t_u, x_s, y_s, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            batch_num = len(x_s)/self.batch_size
            while True:
                (_, c_loss, r_loss, cmd_c_loss, accuracy) = self.sess.run(
                    [self.solver, self.C_loss, self.R_loss, self.cmd_c_loss, self.accuracy_s],
                    feed_dict={self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               self.X_s: x_s,
                               self.Y_s: y_s,
                               }
                )
                print('c_loss:{0}'.format(c_loss))
                print('r_loss:{0}'.format(r_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('accuracy_s:{0}'.format(accuracy))
                if accuracy > 0.7:
                    valid_accuracy, = \
                        self.sess.run([self.accuracy_s, ],
                                      feed_dict={self.X_s: x_valid,
                                                 self.Y_s: y_valid,
                                                 }
                                      )
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best valid result :{0}'.format(best_result))
                        break
                    print('valid_accuracy:{0}'.format(valid_accuracy))
            saver.restore(self.sess, self.model_save_to)
            test_accuracy = self.accuracy_s.eval({self.X_s: x_test, self.Y_s: y_test}, session=self.sess)
            print('Test accuracy:', test_accuracy)
            return test_accuracy


class DCMD(SCMD):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        super(DCMD, self).__init__(source_domain, target_domain, **kwargs)
        self.model_save_to = 'output/model/cmd_extend/DCMD_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.lamb = 1.

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

        def get_correlation(mat_a, mat_b):
            mat_a = mat_a - tf.reduce_mean(mat_a, axis=0)
            mat_b = mat_b - tf.reduce_mean(mat_b, axis=0)
            sigma = tf.matmul(tf.transpose(mat_a), mat_b)
            mat_a_cov = tf.matmul(tf.transpose(mat_a), mat_a)
            mat_b_cov = tf.matmul(tf.transpose(mat_b), mat_b)
            return tf.matmul(tf.matmul(tf.diag(tf.pow(tf.diag_part(mat_a_cov), -0.5)), sigma),
                             tf.diag(tf.pow(tf.diag_part(mat_b_cov), -0.5)))

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

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        self.encoding_c_s_u = encoding_c_s_u
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        self.encoding_c_t_u = encoding_c_t_u
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        self.encoding_t_t_u = encoding_t_t_u
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        self.encoding_t_s_u = encoding_t_s_u
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 3)
        self.corr_loss = tf.reduce_mean(tf.abs(get_correlation(encoding_c_t_u, encoding_t_t_u)))
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u))
        # Get common classification loss
        encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        pred_s = self.common_output_mlp.apply(encoding_c_s)
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s))
        correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters +
                      self.common_output_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.R_loss +
                     self.alpha * self.C_loss +
                     self.gamma * self.cmd_c_loss +
                     self.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))
        self.l2_norm = l2_norm

    def train(self, x_s_u, x_t_u, x_s, y_s, x_valid, y_valid, x_test, y_test, x_s_tst, x_t_tst):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            while True:
                (_, c_loss, r_loss, cmd_c_loss, cmd_t_loss, corr_loss, accuracy) = self.sess.run(
                    [self.solver, self.C_loss, self.R_loss, self.cmd_c_loss, -self.cmd_t_loss, self.corr_loss,
                     self.accuracy_s],
                    feed_dict={self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               self.X_s: x_s,
                               self.Y_s: y_s,
                               }
                )
                print('corr_loss', corr_loss)
                if accuracy > 0.7:
                    valid_accuracy, = \
                        self.sess.run([self.accuracy_s, ],
                                      feed_dict={self.X_s: x_valid,
                                                 self.Y_s: y_valid,
                                                 }
                                      )
                    if valid_accuracy > best_result:
                        best_result = valid_accuracy
                        wait_times = 0
                        print('Save model...')
                        saver.save(self.sess, save_path=self.model_save_to)
                        print('Done')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best valid result :{0}'.format(best_result))
                        break
                    print('valid_accuracy:{0}'.format(valid_accuracy))
            saver.restore(self.sess, self.model_save_to)
            test_accuracy = self.accuracy_s.eval({self.X_s: x_test, self.Y_s: y_test}, session=self.sess)
            print('Test accuracy:', test_accuracy)
            encoding_c_s_u, encoding_c_t_u, encoding_t_s_u, encoding_t_t_u = self.sess.run(
                [self.encoding_c_s_u, self.encoding_c_t_u, self.encoding_t_s_u, self.encoding_t_t_u],
            feed_dict={self.X_s_u: x_s_tst,
                       self.X_t_u: x_t_tst})
            plot_dist(encoding_c_s_u, encoding_c_t_u, 'common_{0}_{1}.pdf'.format(self.source_domain, self.target_domain))
            plot_dist(encoding_t_s_u, encoding_t_t_u, 'target_{0}_{1}.pdf'.format(self.source_domain, self.target_domain))


class TransferClassifier(object):
    '''
    Adapt from source domain to target domain with CMD regularizer
    '''
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.model_save_to = 'output/model/cmd/CMD_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.n_input = 5000
        self.n_hidden_1 = 50
        self.n_classes = 2
        self.d_hidden = 50
        self.tolerate_time = 20
        self.alpha = 2.

    def build_model(self):

        def encoding(x, weights, biases):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            return layer_1

        def predict(x, weights, biases):
            out_layer = tf.matmul(x, weights['out']) + biases['out']
            return out_layer

        def matchnorm(x1, x2):
            return tf.sqrt(tf.reduce_sum(tf.pow(x1 - x2, 2)))
            # return ((x1-x2)**2).sum().sqrt()

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
        self.X_s = tf.placeholder("float", [None, self.n_input])
        self.X_t = tf.placeholder("float", [None, self.n_input])
        self.Y_s = tf.placeholder("float", [None, self.n_classes])
        self.Y_t = tf.placeholder("float", [None, self.n_classes])
        self.X_s_u = tf.placeholder("float", [None, self.n_input])
        self.X_t_u = tf.placeholder("float", [None, self.n_input])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(xavier_init([self.n_input, self.n_hidden_1])),
            'out': tf.Variable(xavier_init([self.n_hidden_1, self.n_classes])),
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros(shape=[self.n_hidden_1])),
            'out': tf.Variable(tf.zeros(shape=[self.n_classes])),
        }

        self.theta = list(self.weights.values()) + list(self.biases.values())

        encoding_s = encoding(self.X_s, self.weights, self.biases)
        self.encoding_s = encoding_s
        self.pred_s = predict(encoding_s, self.weights, self.biases)
        self.accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred_s, 1),
                                                          tf.argmax(self.Y_s, 1)), 'float'))

        encoding_t = encoding(self.X_t, self.weights, self.biases)
        self.encoding_t = encoding_t
        self.pred_t = predict(encoding_t, self.weights, self.biases)
        self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred_t, 1),
                                                          tf.argmax(self.Y_t, 1)), 'float'))

        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_s,
                                                                             labels=self.Y_s))
        self.encoding_s_u = encoding(self.X_s_u, self.weights, self.biases)
        self.encoding_t_u = encoding(self.X_t_u, self.weights, self.biases)
        self.D_loss = -mmatch(self.encoding_s_u, self.encoding_t_u, 3)
        self.l2_norm = 0.
        for tensor in list(self.weights.values()):
            self.l2_norm += tf.reduce_sum(tf.abs(tensor))
        self.loss = self.C_loss + self.alpha * self.D_loss + 0.0001*self.l2_norm
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))
        self.C_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                         .minimize(self.C_loss + .0001 * self.l2_norm, var_list=self.theta))

    def train(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            wait_times = 0
            best_result = 0.
            while True:
                _, loss, C_loss, D_loss, accuracy = self.sess.run(
                    [self.solver, self.loss, self.C_loss, self.D_loss, self.accuracy_s],
                    feed_dict={self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               self.X_s: x_s,
                               self.Y_s: y_s,
                               }
                )
                print('total_loss:{0}'.format(loss))
                print('C_loss:{0}'.format(C_loss))
                print('D_loss:{0}'.format(D_loss))
                print('accuracy:{0}'.format(accuracy))
                # Do validation
                if accuracy > 0.7:
                    test_accuracy = self.accuracy_t.eval({self.X_t: x_valid, self.Y_t: y_valid}, session=self.sess)
                    if test_accuracy > best_result:
                        best_result = test_accuracy
                        wait_times = 0
                        # print('save model...')
                        saver.save(self.sess, self.model_save_to)
                        # print('done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("Valid accuracy:", test_accuracy)
            saver.restore(self.sess, self.model_save_to)
            # test_accuracy = self.accuracy_t.eval({self.X_t: x_test, self.Y_t: y_test}, session=self.sess)
            # print("Test accuracy:", test_accuracy)
            encoding_s, encoding_t = self.sess.run(
                [self.encoding_s_u, self.encoding_t_u],
                feed_dict={self.X_s_u: x_s_u,
                           self.X_t_u: x_t_u})
            print(encoding_s.shape)
            common_a_distance = plot_dist(encoding_s, encoding_t, '{0}_{1}.pdf'.format(self.source_domain, self.target_domain))
            print('common_a_distance', common_a_distance)
            # return test_accuracy


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
        self.common_tune_model_save_to = 'output/model/cmd_extend/Cotrain_common_tune_{0}_to_{1}.pkl'.format(source_domain, target_domain)
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

        def get_correlation(mat_a, mat_b):
            mat_a = mat_a - tf.reduce_mean(mat_a, axis=0)
            mat_b = mat_b - tf.reduce_mean(mat_b, axis=0)
            sigma = tf.matmul(tf.transpose(mat_a), mat_b)
            mat_a_cov = tf.matmul(tf.transpose(mat_a), mat_a)
            mat_b_cov = tf.matmul(tf.transpose(mat_b), mat_b)
            return tf.matmul(tf.matmul(tf.diag(tf.pow(tf.diag_part(mat_a_cov), -0.5)), sigma),
                             tf.diag(tf.pow(tf.diag_part(mat_b_cov), -0.5)))

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
        # Get correlation loss
        self.corr_loss = tf.reduce_mean(tf.abs(get_correlation(encoding_c_t_u, encoding_t_t_u)))
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
                     # self.alpha * self.C_loss +
                     # self.belta * self.T_loss +
                     self.gamma * self.cmd_c_loss +
                     self.lamb * self.cmd_t_loss +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))

        self.common_theta = (self.common_encode_mlp.parameters +
                             self.common_decode_mlp.parameters +
                             self.common_output_mlp.parameters)
        self.common_loss = (self.R_loss +
                            self.alpha * self.C_loss +
                            self.gamma * self.cmd_c_loss +
                            self.corr_loss +
                            0.0001 * l2_norm)
        self.common_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.common_loss, var_list=self.common_theta))
        self.target_theta = (self.target_encode_mlp.parameters +
                             # self.target_decode_mlp.parameters +
                             self.target_output_mlp.parameters)
        self.target_loss = (self.R_loss +
                            self.belta * self.T_loss +
                            self.lamb * self.cmd_t_loss +
                            self.corr_loss +
                            0.0001 * l2_norm)
        self.target_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.target_loss, var_list=self.target_theta))
        self.combined_loss = (self.C_loss + 0.0001 * l2_norm)
        self.combined_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                              .minimize(self.combined_loss, var_list=self.theta))

    def initialize_model(self, x_s_u, x_t_u, x_s, y_s, x_t, y_t, x_valid, y_valid, x_test, y_test):
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
            for _ in range(150):
                (_, c_loss, cmd_c_loss, cmd_t_loss, r_loss, accuracy) = self.sess.run(
                    [self.solver, self.C_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss,
                     self.accuracy_s],
                    feed_dict={self.X_s: x_s,
                               self.Y_s: y_s,
                               self.X_t: x_t,
                               self.Y_t: y_t,
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
                        saver.save(self.sess, save_path=self.target_model_save_to)
                        print('Done!')
                    else:
                        wait_times += 1
                    if wait_times >= self.tolerate_time:
                        print('best_result:{0}'.format(best_result))
                        break
                    print("valid accuracy:", valid_accuracy)
            saver.save(self.sess, save_path=self.target_model_save_to)
            return best_result

    def train_common_model(self, x_s_u, x_t_u, x_t, y_t, x_valid, y_valid, x_test, y_test):
        wait_times = 0
        best_result = 0.
        self.graph = tf.Graph()
        tfConfig = tf.ConfigProto()
        tfConfig.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.graph, config=tfConfig)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.target_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.target_model_save_to)
            saver = tf.train.Saver(var_list=self.theta)
            while True:
                (_, c_loss, cmd_c_loss, r_loss, corr_loss, accuracy) = self.sess.run(
                    [self.common_solver, self.C_loss, self.cmd_c_loss, self.R_loss, self.corr_loss,
                     self.accuracy_s],
                    feed_dict={self.X_s: x_t,
                               self.Y_s: y_t,
                               self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('c_loss:{0}'.format(c_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('r_loss:{0}'.format(r_loss))
                print('corr_loss:{0}'.format(corr_loss))
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
            saver = tf.train.Saver(var_list=self.common_theta)
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, self.common_model_save_to)
            saver = tf.train.Saver(var_list=self.theta)
            while True:
                (_, t_loss, cmd_c_loss, cmd_t_loss,
                 r_loss, corr_loss, accuracy) = self.sess.run(
                    [self.target_solver, self.T_loss, self.cmd_c_loss, -self.cmd_t_loss, self.R_loss, self.corr_loss, self.accuracy_t],
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
                print('corr_loss:{0}'.format(corr_loss))
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
        self.initialize_model(x_s_u, x_t_u, x_s, y_s, x_t, y_t,
                              x_valid, y_valid, x_test, y_test)
        wait_times = 0.
        while len(U) > 0:
            print('Train common model...')
            self.train_common_model(x_s_u, x_t_u, np.concatenate([x_s, x_t]), np.concatenate([y_s, y_t]),
                                    x_valid, y_valid, x_test, y_test)
            # input = raw_input('Input andy character!'
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
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==0.], H_t_hat[:, 1][y_t_train[:, 0]==0.], color='r', alpha=.4, s=1)
        plt.scatter(H_t_hat[:, 0][y_t_train[:, 0]==1.], H_t_hat[:, 1][y_t_train[:, 0]==1.], color='b', alpha=.4, s=1)
        plt.savefig("h_t.pdf", dpi=72)


class Autoencoder(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 5e-3
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd_extend/Auto_{0}_to_{1}.pkl'.format(source_domain, target_domain)
        self.model_load_from = self.model_save_to
        self.n_input = 5000
        self.n_classes = 2
        self.model_built = False
        self.tolerate_time = 20
        self.alpha = 1.
        self.belta = 1.
        self.gamma = 1.
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

        encoding_c_s_u = self.common_encode_mlp.apply(self.X_s_u)
        self.encoding_c_s_u = encoding_c_s_u
        encoding_c_t_u = self.common_encode_mlp.apply(self.X_t_u)
        self.encoding_c_t_u = encoding_c_t_u
        encoding_t_s_u = self.target_encode_mlp.apply(self.X_s_u)
        self.encoding_t_s_u = encoding_t_s_u
        encoding_t_t_u = self.target_encode_mlp.apply(self.X_t_u)
        self.encoding_t_t_u = encoding_t_t_u
        # Get cmd loss
        self.cmd_c_loss = mmatch(encoding_c_s_u, encoding_c_t_u, 3)
        self.cmd_t_loss = -mmatch(encoding_t_s_u, encoding_t_t_u, 3)
        # Get reconstruction loss
        decoding_c_t_u = self.common_decode_mlp.apply(encoding_c_t_u)
        decoding_t_t_u = self.target_decode_mlp.apply(encoding_t_t_u)
        decoding_c_s_u = self.common_decode_mlp.apply(encoding_c_s_u)
        decoding_t_s_u = self.target_decode_mlp.apply(encoding_t_s_u)
        decoding_t_u = decoding_c_t_u + decoding_t_t_u
        decoding_s_u = decoding_c_s_u + decoding_t_s_u
        self.R_loss = tf.reduce_mean(tf.square(decoding_t_u - self.X_t_u)) + tf.reduce_mean(tf.square(decoding_s_u - self.X_s_u))
        # Get classification loss
        # encoding_c_s = self.common_encode_mlp.apply(self.X_s)
        # pred_s = self.common_output_mlp.apply(encoding_c_s)
        # self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_s, labels=self.Y_s))
        # correct_prediction = tf.equal(tf.argmax(pred_s, 1), tf.argmax(self.Y_s, 1))
        # self.accuracy_s = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Build solver
        # Build solver
        self.theta = (self.common_encode_mlp.parameters +
                      self.target_encode_mlp.parameters +
                      self.common_decode_mlp.parameters +
                      self.target_decode_mlp.parameters)
        l2_norm = 0.
        for tensor in self.theta:
            if tensor.name.find('W') != 0:
                l2_norm += tf.reduce_sum(tf.abs(tensor))
        self.loss = (self.R_loss +
                     self.gamma * (self.cmd_c_loss + self.cmd_t_loss) +
                     0.0001 * l2_norm)
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                       .minimize(self.loss, var_list=self.theta))

    def train(self, x_s_u, x_t_u):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.build_model()
            saver = tf.train.Saver(var_list=self.theta)
            self.sess.run(tf.global_variables_initializer())
            for _ in range(500):
                (_, r_loss, cmd_c_loss, cmd_t_loss) = self.sess.run(
                    [self.solver, self.R_loss, self.cmd_c_loss, -self.cmd_t_loss],
                    feed_dict={self.X_s_u: x_s_u,
                               self.X_t_u: x_t_u,
                               }
                )
                print('r_loss:{0}'.format(r_loss))
                print('cmd_c_loss:{0}'.format(cmd_c_loss))
                print('cmd_t_loss:{0}'.format(cmd_t_loss))
            # saver.save(self.sess, self.model_save_to)
            encoding_s, encoding_t = self.sess.run(
                [self.encoding_c_s_u, self.encoding_c_t_u],
                feed_dict={self.X_s_u: x_s_u,
                           self.X_t_u: x_t_u})
            plot_dist(encoding_s, encoding_t,
                                          'auto_{0}_{1}.pdf'.format(self.source_domain, self.target_domain))
            return None


if __name__ == '__main__':
    data_load_from = 'data/cmd/amazon.mat'
    import sys
    mean_results = []
    for source_domain in [0, 2]:
        for target_domain in [1, 3]:
            if source_domain == target_domain:
                continue
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

            x_t_tune = np.copy(x_t_tst[:50])
            y_t_tune = np.copy(y_t_tst[:50])
            x_t_tst = x_t_tst[50:]
            y_t_tst = y_t_tst[50:]

            x_t_valid = x_t_tst[:500]
            y_t_valid = y_t_tst[:500]
            x_t_tst = x_t_tst[500:]
            y_t_tst = y_t_tst[500:]


            classifier = SourceOnly(source_domain, target_domain)
            classifier.train(x_s_tr, y_s_tr, x_t_valid, y_t_valid, x_t_tr, y_t_tr, x_s_tst, x_t_tst)
            classifier = DCMD(source_domain, target_domain)
            classifier.train(x_s_tr, x_t_tr, x_s_tr, y_s_tr,
                      x_t_valid, y_t_valid, x_t_tst, y_t_tst, x_s_tst, x_t_tst)

            # input = raw_input('Input to continue!')
            # Fine-tune with tiny target domain samples on source domain model
            # print('Fine-tune with tiny target domain samples on source domain model')
            # cmd_results = []
            # fine_tune_results = []
            # cmd_classifier = TransferClassifier(source_domain, target_domain)
            # classifier = FineTuneClassifier(source_domain, target_domain)
            # for i in range(5):
            #     cmd_result = cmd_classifier.train(x_s_tr, x_t_tr, x_s_tr, y_s_tr, x_t_tune, y_t_tune,
            #           x_t_valid, y_t_valid, x_t_tst, y_t_tst)
            #     # input = raw_input('Input any character!')
            #     fine_tune_result = classifier.train(x_s_tr, x_t_tr, x_s_tr, y_s_tr, x_t_tune, y_t_tune,
            #                                     x_t_valid, y_t_valid, x_t_tst, y_t_tst)
            #     # input = raw_input('Input any character!')
            #     cmd_results.append(cmd_result)
            #     fine_tune_results.append(fine_tune_result)
            # print('Source domain:{0}\t Target domain:{1}'.format(source_domain, target_domain))
            # print('cmd_result:', sum(cmd_results)/len(cmd_results))
            # print('fine_tune_result:', sum(fine_tune_results)/len(fine_tune_results))


    for result in mean_results:
        print(result)
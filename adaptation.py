import tensorflow as tf
import numpy as np
from utils_amazon import load_amazon, split_data, identity


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def shuffle(arrays):
    """
    shuffle data (used by split)
    """
    index_shuf = np.arange(arrays[0].shape[0])
    np.random.shuffle(index_shuf)
    return [array[index_shuf] for array in arrays]


class BaseAdaptation(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.learning_rate = 1e-5
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd/{0}_to_{1}.pkl'.format(source_domain, target_domain)
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
        # Extract information with autoencoder and train source domain classifier on extracted information
        self.model_built = True
        n_hidden_1 = self.n_hidden_1
        n_classes = self.n_classes

        def encoding(x, weight, bias):
            layer_1 = tf.add(tf.matmul(x, weight), bias)
            layer_1 = tf.nn.sigmoid(layer_1)
            return layer_1

        def decoding(x, weight, bias):
            output = tf.nn.relu(tf.add(tf.matmul(x, weight), bias))
            return output

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
        self.Y_s = tf.placeholder("float", [None, n_classes])
        self.Y_t = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = {
            'h_e': tf.Variable(xavier_init([self.n_input, n_hidden_1])),
            'h_d': tf.Variable(xavier_init([self.n_hidden_1, self.n_input])),
        }
        self.biases = {
            'b_e': tf.Variable(tf.zeros(shape=[n_hidden_1])),
            'b_d': tf.Variable(tf.zeros(shape=[self.n_input])),
        }

        self.theta_A = [self.weights['h_e'], self.weights['h_d'], self.biases['b_e'], self.biases['b_d']]

        # Autoencoder
        self.encoding_s = encoding(self.X_s, self.weights['h_e'], self.biases['b_e'])
        self.decoding_s = decoding(self.encoding_s, self.weights['h_d'], self.biases['b_d'])

        self.encoding_t = encoding(self.X_t, self.weights['h_e'], self.biases['b_e'])
        self.decoding_t = decoding(self.encoding_t, self.weights['h_d'], self.biases['b_d'])

        self.R_loss = (tf.reduce_mean(tf.square(self.decoding_s - self.X_s)) +
                       tf.reduce_mean(tf.square(self.decoding_t - self.X_t)))   # Reconstruction loss

        self.D_loss = mmatch(self.encoding_s, self.encoding_t, 5)               # Distribution loss

        self.A_loss = self.R_loss + self.alpha * self.D_loss                    # Auto-encode loss
        self.A_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate * 50)
                       .minimize(self.A_loss, var_list=self.theta_A))
        # Build classifier on encoded feature representation
        self.mlp = MLP(name='output_mlp', dims=[self.n_hidden_1, self.n_hidden_1, n_classes],
                  activations=[tf.nn.relu, identity])
        self.theta_C = self.mlp.parameters
        self.pred_s = self.mlp.apply(self.encoding_s)
        self.pred_t = self.mlp.apply(self.encoding_t)
        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_s, labels=self.Y_s))
        self.C_solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate * 50)
                       .minimize(self.C_loss, var_list=self.theta_C))

    def train(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.build_model()
            batch_size = self.batch_size
            x, y, offset = load_amazon(5000, self.data_load_from)

            x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(self.source_domain,
                                                                                            self.target_domain,
                                                                                            x, y, offset, 2000)
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            # Do auto-encoding
            for epoch in range(500):
                A_loss_num = 0.
                R_loss_num = 0.
                D_loss_num = 0.
                _, A_loss_curr, R_loss_curr, D_loss_curr = self.sess.run(
                    [self.A_solver, self.A_loss, self.R_loss, self.D_loss],
                    feed_dict={self.X_s: x_s_tr[i * batch_size:(i + 1) * batch_size, ],
                               self.X_t: x_t_tr[i * batch_size:(i + 1) * batch_size, ],
                               }
                )
                A_loss_num += A_loss_curr
                R_loss_num += R_loss_curr
                D_loss_num += D_loss_curr
                x_s_tr, y_s_tr = shuffle([x_s_tr, y_s_tr])
                x_t_tr, y_t_tr = shuffle([x_t_tr, y_t_tr])
                batch_num = len(x_s_tr) / batch_size
                print('A_loss:{0}'.format(A_loss_num / batch_num))
                print('R_loss:{0}'.format(R_loss_num / batch_num))
                print('D_loss:{0}'.format(D_loss_num / batch_num))
            # Do classification
            wait_times = 0
            best_result = 0.
            for epoch in range(500):
                total_loss = 0.
                C_loss_num = 0.
                D_loss_num = 0.
                accuracy_num = 0.

                for i in range(len(x_s_tr) / batch_size):
                    _, loss_curr, C_loss_curr, D_loss_curr, accuracy = self.sess.run(
                        [self.solver, self.loss, self.C_loss, self.D_loss, self.accuracy_s],
                        feed_dict={self.X_s: x_s_tr[i * batch_size:(i + 1) * batch_size, ],
                                   self.X_t: x_t_tr[i * batch_size:(i + 1) * batch_size, ],
                                   self.Y_s: y_s_tr[i * batch_size:(i + 1) * batch_size, ]}
                    )
                    total_loss += loss_curr
                    D_loss_num += D_loss_curr
                    C_loss_num += C_loss_curr
                    accuracy_num += accuracy
                x_s_tr, y_s_tr = shuffle([x_s_tr, y_s_tr])
                x_t_tr, y_t_tr = shuffle([x_t_tr, y_t_tr])
                batch_num = len(x_s_tr) / batch_size
                print('total_loss:{0}'.format(total_loss / batch_num))
                print('C_loss:{0}'.format(C_loss_num / batch_num))
                print('D_loss:{0}'.format(D_loss_num / batch_num))
                print('train_accuracy:{0}'.format(accuracy_num / batch_num))
                # Temporarily valid on test set
                test_accuracy = self.accuracy_t.eval({self.X_t: x_t_tst, self.Y_t: y_t_tst})
                if test_accuracy > best_result:
                    best_result = test_accuracy
                    wait_times = 0
                    print('save model...')
                    saver.save(self.sess, self.model_save_to)
                    print('done!')
                else:
                    wait_times += 1
                if wait_times >= self.tolerate_time:
                    print('best_result:{0}'.format(best_result))
                    break
                print("Test accuracy:", test_accuracy)

    def get_hidden_state(self, X):
        if self.sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)

            with self.graph.as_default():
                self.build_model()
                saver = tf.train.Saver()
                try:
                    saver.restore(self.sess, self.model_load_from)
                except:
                    self.train()
                    saver.restore(self.sess, self.model_save_to)
        hidden, = self.sess.run([self.encoding_s], feed_dict={self.X_s: X})
        return hidden

    def get_prediction(self, X):
        if self.sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)

            with self.graph.as_default():
                self.build_model()
                saver = tf.train.Saver()
                try:
                    saver.restore(self.sess, self.model_load_from)
                except:
                    self.train()
                    saver.restore(self.sess, self.model_save_to)
        prediction, = self.sess.run([self.pred_s], feed_dict={self.X_s: X})
        return prediction


class CMDAdaptation(object):
    def __init__(self, source_domain=0, target_domain=3, **kwargs):
        self.source_domain =source_domain
        self.target_domain = target_domain
        self.learning_rate = 1e-5
        self.data_load_from = 'data/cmd/amazon.mat'
        self.batch_size = 200
        self.model_save_to = 'output/model/cmd/{0}_to_{1}.pkl'.format(source_domain, target_domain)
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
        self.Y_s = tf.placeholder("float", [None, n_classes])
        self.Y_t = tf.placeholder("float", [None, n_classes])

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

        # Construct model
        self.encoding_s = encoding(self.X_s, self.weights, self.biases)
        self.pred_s = predict(self.encoding_s, self.weights, self.biases)
        self.accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred_s, 1),
                                                              tf.argmax(self.Y_s, 1)), 'float'))

        self.encoding_t = encoding(self.X_t, self.weights, self.biases)
        self.pred_t = predict(self.encoding_t, self.weights, self.biases)
        self.accuracy_t = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred_t, 1),
                                                              tf.argmax(self.Y_t, 1)), 'float'))

        self.C_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred_s,
                                                                             labels=self.Y_s))
        self.D_loss = mmatch(self.encoding_s, self.encoding_t, 5)
        self.loss = self.C_loss + self.alpha * self.D_loss
        self.solver = (tf.train.RMSPropOptimizer(learning_rate=self.learning_rate * 50)
                  .minimize(self.loss, var_list=self.theta))

    def train(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            self.build_model()
            batch_size = self.batch_size
            x, y, offset = load_amazon(5000, self.data_load_from)

            # Launch the graph
            x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(self.source_domain,
                                                                                            self.target_domain,
                                                                                            x, y, offset, 2000)
            # Try to initialize model
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            wait_times = 0
            best_result = 0.
            while True:
                total_loss = 0.
                C_loss_num = 0.
                D_loss_num = 0.
                accuracy_num = 0.

                for i in range(len(x_s_tr) / batch_size):
                    _, loss_curr, C_loss_curr, D_loss_curr, accuracy = self.sess.run(
                        [self.solver, self.loss, self.C_loss, self.D_loss, self.accuracy_s],
                        feed_dict={self.X_s: x_s_tr[i * batch_size:(i + 1) * batch_size, ],
                                   self.X_t: x_t_tr[i * batch_size:(i + 1) * batch_size, ],
                                   self.Y_s: y_s_tr[i * batch_size:(i + 1) * batch_size, ]}
                    )
                    total_loss += loss_curr
                    D_loss_num += D_loss_curr
                    C_loss_num += C_loss_curr
                    accuracy_num += accuracy
                x_s_tr, y_s_tr = shuffle([x_s_tr, y_s_tr])
                x_t_tr, y_t_tr = shuffle([x_t_tr, y_t_tr])
                batch_num = len(x_s_tr) / batch_size
                print('total_loss:{0}'.format(total_loss / batch_num))
                print('C_loss:{0}'.format(C_loss_num / batch_num))
                print('D_loss:{0}'.format(D_loss_num / batch_num))
                print('train_accuracy:{0}'.format(accuracy_num / batch_num))
                # Temporarily valid on test set
                test_accuracy = self.accuracy_t.eval({self.X_t: x_t_tst, self.Y_t: y_t_tst},
                                                     session=self.sess)
                if test_accuracy > best_result:
                    best_result = test_accuracy
                    wait_times = 0
                    print('save model...')
                    saver.save(self.sess, self.model_save_to)
                    print('done!')
                else:
                    wait_times += 1
                if wait_times >= self.tolerate_time:
                    print('best_result:{0}'.format(best_result))
                    break
                print("Test accuracy:", test_accuracy)

    def get_hidden_state(self, X):
        if self.sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
            with self.graph.as_default():
                self.build_model()
                saver = tf.train.Saver()
                try:
                    saver.restore(self.sess, self.model_load_from)
                except:
                    self.train()
                    saver.restore(self.sess, self.model_save_to)
        hidden, = self.sess.run([self.encoding_s], feed_dict={self.X_s: X})
        return hidden

    def get_prediction(self, X):
        if self.sess is None:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)

            with self.graph.as_default():
                self.build_model()
                saver = tf.train.Saver()
                try:
                    saver.restore(self.sess, self.model_load_from)
                except:
                    self.train()
                    saver.restore(self.sess, self.model_save_to)
        prediction, = self.sess.run([self.pred_s], feed_dict={self.X_s: X})
        # It is better to use non-softmax value
        # prediction = np.exp(prediction-prediction.max(axis=1, keepdims=True))
        # prediction /= prediction.sum(axis=1, keepdims=True)
        return prediction


if __name__ == '__main__':
    adaptation = CMDAdaptation()
    adaptation.train()
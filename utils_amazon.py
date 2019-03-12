import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

import tensorflow as tf


def load_amazon(n_features, filename):
    """
    Load amazon reviews
    """
    mat = loadmat(filename)
    
    xx=mat['xx']
    yy=mat['yy']
    offset=mat['offset']
    
    x=xx[:n_features,:].toarray().T#n_samples X n_features
    y=yy.ravel()
    
    return x, y, offset

def shuffle(x, y):
    """
    shuffle data (used by split)
    """
    index_shuf = np.arange(x.shape[0])
    np.random.shuffle(index_shuf)
    x=x[index_shuf,:]
    y=y[index_shuf]
    return x,y

def to_one_hot(a):
    b = np.zeros((len(a), 2))
    b[np.arange(len(a)), a] = 1
    return b

def split_data(d_s_ind,d_t_ind,x,y,offset,n_tr_samples,r_seed=0):

    # x = normalize(x, axis=0, norm='max')
    # x = np.log(1.+x)
    np.random.seed(r_seed)
    x_s_tr = x[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples,:]
    x_t_tr = x[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples,:]
    x_s_tst = x[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0],:]
    x_t_tst = x[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0],:]
    y_s_tr = y[offset[d_s_ind,0]:offset[d_s_ind,0]+n_tr_samples]
    y_t_tr = y[offset[d_t_ind,0]:offset[d_t_ind,0]+n_tr_samples]
    y_s_tst = y[offset[d_s_ind,0]+n_tr_samples:offset[d_s_ind+1,0]]
    y_t_tst = y[offset[d_t_ind,0]+n_tr_samples:offset[d_t_ind+1,0]]
    x_s_tr,y_s_tr=shuffle(x_s_tr,y_s_tr)
    x_t_tr,y_t_tr=shuffle(x_t_tr,y_t_tr)
    x_s_tst,y_s_tst=shuffle(x_s_tst,y_s_tst)
    x_t_tst,y_t_tst=shuffle(x_t_tst,y_t_tst)
    y_s_tr[y_s_tr==-1]=0
    y_t_tr[y_t_tr==-1]=0
    y_s_tst[y_s_tst==-1]=0
    y_t_tst[y_t_tst==-1]=0

    y_s_tr=to_one_hot(y_s_tr)
    y_t_tr=to_one_hot(y_t_tr)
    y_s_tst=to_one_hot(y_s_tst)
    y_t_tst=to_one_hot(y_t_tst)

    return x_s_tr,y_s_tr,x_t_tr,y_t_tr,x_s_tst,y_s_tst,x_t_tst,y_t_tst

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def turn_tfidf(x):
    df = (x > 0.).sum(axis=0)
    idf = np.log(1.* len(x)/(df+1))
    return np.log(1.+x) * idf[None, :]

def turn_one_hot(x):
    return (x > 0.).astype('float32')

def identity(x):
    return x


class MLP(object):
    def __init__(self, name, dims, activations):
        self.name = name
        self.dims = dims
        self.activations = activations
        self.weights = []
        self.biases = []
        self._initialize()

    @property
    def parameters(self):
        return self.weights + self.biases

    def _initialize(self):
        for i in range(len(self.dims)-1):
            w = tf.Variable(xavier_init([self.dims[i], self.dims[i+1]]), name=self.name+'_W_{0}'.format(i))
            b = tf.Variable(xavier_init([self.dims[i+1]]), name=self.name+'_b_{0}'.format(i))
            self.weights.append(w)
            self.biases.append(b)

    def apply(self, x):
        out = x
        for activation, weight, bias in zip(self.activations, self.weights, self.biases):
            out = activation(tf.add(tf.matmul(out, weight), bias))
        return out


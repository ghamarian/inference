import tensorflow as tf
import numpy as np
import pandas as pd

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS


def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction * n)
        n_train = n - n_valid
        I = np.random.permutation(range(0, n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid


def log(logfile, str):
    """ Log a string in a file """
    with open(logfile, 'a') as f:
        f.write(str + '\n')
    print str


def save_config(fname):
    """ Save configuration """
    flagdict = FLAGS.__dict__['__flags']
    s = '\n'.join(['%s: %s' % (k, str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname, 'w')
    f.write(s)
    f.close()


class DataSet:

    def __init__(self, fname):
        self.data_in = np.load(fname)

        self.df_x, self.n, self.dim, self.num_experiments = self.load_x()
        self.df_y = self.load_y()

        self.have_truth = 'ycf' in self.data_in
        self.list = self.creat_list()

    def load_x(self):
        x = np.swapaxes(self.data_in['x'], 0, 2)
        n, dim, num_experiments = x.shape
        experiments = np.swapaxes(x, 1, 2)

        df_x = pd.Panel(experiments).to_frame()
        df_x.columns.name = 'exper'
        df_x.index = df_x.index.rename(['row', 'column'])

        # Make experiment to be the first level index
        df_x = df_x.stack(0).unstack([0, 1]).stack(0)

        return df_x, n, dim, num_experiments

    def load_y(self):
        column_dict = {k: (self.data_in[k].T if k in self.data_in else np.empty_like((self.num_experiments, self.n)))
                       for k in ['t', 'yf', 'ycf']}

        df_y = pd.Panel(column_dict).to_frame()
        df_y.index = df_y.index.rename(['expr', 'row'])

        return df_y

    def creat_list(self, ):
        return [self.load_experiment_dataset(i) for i in range(self.num_experiments)]

    def get_x_of(self, i_exp):
        return self.df_x.xs(i_exp).values

    def get_labels_of(self, i_exp):
        result = self.df_y.xs(i_exp).T
        result = dict(zip(result.index, result.values[:, :, None]))
        return result

    def load_experiment_dataset(self, i_exp):
        result = {'x': self.get_x_of(i_exp)}
        result.update(self.get_labels_of(i_exp))

        result['HAVE_TRUTH'] = self.have_truth
        return result

    def __getitem__(self, item):
        return self.list[item]


def load_data_df(fname):
    # if fname[-3:] == 'npz':
    dataset = DataSet(fname)
    return dataset


# def load_data(fname):
#     """ Load data set """
#     if fname[-3:] == 'npz':
#         data_in = np.load(fname)
#         data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
#         # data = {iKey: data_in[iKey] for iKey in ['x', 't', 'yf']}
#
#         data['ycf'] = data_in['ycf'] if 'ycf' in data_in else None
#
#     else:
#         if FLAGS.sparse>0:
#             data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
#             x = load_sparse(fname+'.x')
#         else:
#             data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
#             x = data_in[:,5:]
#
#         data = {}
#         data['x'] = x
#         data['t'] = data_in[:,0:1]
#         data['yf'] = data_in[:,1:2]
#         data['ycf'] = data_in[:,2:3]
#
#     data['HAVE_TRUTH'] = not data['ycf'] is None
#
#     data['dim'] = data['x'].shape[1]
#     data['n'] = data['x'].shape[0]
#
#     return data

def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname, "rb"), delimiter=",")
    H = E[0, :]
    n = int(H[0])
    d = int(H[1])
    E = E[1:, :]
    S = sparse.coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
    S = S.todense()

    return S


def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def lindisc(X, p, t):
    ''' Linear MMD '''

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    mean_control = tf.reduce_mean(Xc, reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt, reduction_indices=0)

    c = tf.square(2 * p - 1) * 0.25
    f = tf.sign(p - 0.5)

    mmd = tf.reduce_sum(tf.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + safe_sqrt(c + mmd)

    return mmd


def mmd2_lin(X, t, p):
    ''' Linear MMD '''

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    mean_control = tf.reduce_mean(Xc, reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt, reduction_indices=0)

    mmd = tf.reduce_sum(tf.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

    return mmd


def mmd2_rbf(X, t, p, sig):
    """ Computes the l2-RBF MMD for X given t """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    Kcc = tf.exp(-pdist2sq(Xc, Xc) / tf.square(sig))
    Kct = tf.exp(-pdist2sq(Xc, Xt) / tf.square(sig))
    Ktt = tf.exp(-pdist2sq(Xt, Xt) / tf.square(sig))

    m = tf.to_float(tf.shape(Xc)[0])
    n = tf.to_float(tf.shape(Xt)[0])

    mmd = tf.square(1.0 - p) / (m * (m - 1.0)) * (tf.reduce_sum(Kcc) - m)
    mmd = mmd + tf.square(p) / (n * (n - 1.0)) * (tf.reduce_sum(Ktt) - n)
    mmd = mmd - 2.0 * p * (1.0 - p) / (m * n) * tf.reduce_sum(Kct)
    mmd = 4.0 * mmd

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X, Y))


def pop_dist(X, t):
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt, Xc)
    return M


def wasserstein(X, t, p, lam=10, its=10, sq=False, backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt, Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt, Xc))

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat(0, [delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))])
    Mt = tf.concat(0, [M, row])
    Mt = tf.concat(1, [Mt, col])

    ''' Compute marginal vectors '''
    a = tf.concat(0, [p * tf.ones(tf.shape(tf.where(t > 0)[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))])
    b = tf.concat(0, [(1 - p) * tf.ones(tf.shape(tf.where(t < 1)[:, 0:1])) / nc, p * tf.ones((1, 1))])

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    if not backpropT:
        T = tf.stop_gradient(T)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    return D, Mlam


def simplex_project(x, k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x, axis=0)[::-1]
    nu = (np.cumsum(mu) - k) / range(1, d + 1)
    I = [i for i in range(0, d) if mu[i] > nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x - theta, 0)
    return w

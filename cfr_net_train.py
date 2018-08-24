import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback
import itertools
import pandas as pd

import cfr.cfr_net as cfr
from cfr.util import *

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.app.flags.DEFINE_float('p_alpha', 1e-4, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 10, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none',
                           """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/', """Data directory. """)
# tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('dataform', '../data/ihdp_npci_1-100.test.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_string('imb_fun', 'mmd_lin',
                           """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('output_csv', 0, """Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1,
                            """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1,
                            """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True


class SessionRunner:
    def __init__(self, CFR):
        self.CFR = CFR
        self.sess = tf.Session()

    def create_main_loss_feed_dict(self, train, t_i_train_, cfr_y, expr_dataset):
        dict_cfactual = {self.CFR.x: expr_dataset['x'][train, :],
                         self.CFR.t: t_i_train_,
                         self.CFR.y_: cfr_y,
                         self.CFR.do_in: 1.0,
                         self.CFR.do_out: 1.0}
        return dict_cfactual

    def initialize_global_variables(self):
        self.sess.run(tf.global_variables_initializer())

    def create_loss_feed_dict(self, p_treated, train_or_valid, expr_dataset):
        result = self.create_main_loss_feed_dict(train_or_valid, expr_dataset['t'][train_or_valid, :],
                                                 expr_dataset['yf'][train_or_valid, :], expr_dataset)

        result.update({self.CFR.r_alpha: FLAGS.p_alpha, self.CFR.r_lambda: FLAGS.p_lambda, self.CFR.p_t: p_treated})
        return result

    def set_feed_dicts(self, expr_dataset, p_treated, I_train, I_valid):
        self.dict_factual = self.create_loss_feed_dict(p_treated, I_train, expr_dataset)

        if FLAGS.val_part > 0:
            self.dict_valid = self.create_loss_feed_dict(p_treated, I_valid, expr_dataset)

        if expr_dataset['HAVE_TRUTH']:
            self.dict_cfactual = self.create_main_loss_feed_dict(I_train, 1 - expr_dataset['t'][I_train, :],
                                                                 expr_dataset['ycf'][I_train, :], expr_dataset)

    def _run_losses(self, dict_factual_or_cfactual):
        obj_loss, f_error, imb_err = self.sess.run([self.CFR.tot_loss, self.CFR.pred_loss, self.CFR.imb_dist],
                                                   feed_dict=dict_factual_or_cfactual)

        return obj_loss, f_error, imb_err

    def run_factual_losses(self):
        return self._run_losses(self.dict_factual)

    def run_counter_factual_losses(self):
        return self._run_losses(self.dict_cfactual)

    def run_valid_losses(self):
        return self._run_losses(self.dict_valid)

    def run_pred_loss(self):
        cf_error = self.sess.run(self.CFR.pred_loss, feed_dict=self.dict_cfactual)
        return cf_error

    def run_projection(self, wip):
        return self.sess.run(self.CFR.projection, feed_dict={self.CFR.w_proj: wip})

    def run_h_norm(self, cfr_x, do_in):
        rep = self.sess.run(self.CFR.h_rep_norm, feed_dict={self.CFR.x: cfr_x, self.CFR.do_in: do_in})
        return rep

    def run_output(self, x_batch, t_batch, do_in, do_out):
        y_pred = self.sess.run(self.CFR.output,
                               feed_dict={self.CFR.x: x_batch, self.CFR.t: t_batch, self.CFR.do_in: do_in,
                                          self.CFR.do_out: do_out})

        return y_pred

    def run_train_step(self, train_step, x_batch, t_batch, y_batch, p_treated):
        self.sess.run(train_step, feed_dict={self.CFR.x: x_batch, self.CFR.t: t_batch, self.CFR.y_: y_batch,
                                             self.CFR.do_in: FLAGS.dropout_in,
                                             self.CFR.do_out: FLAGS.dropout_out, self.CFR.r_alpha: FLAGS.p_alpha,
                                             self.CFR.r_lambda: FLAGS.p_lambda, self.CFR.p_t: p_treated})

    def run_descriptive_stats(self, x_batch, t_batch):
        m_statistics = self.sess.run(cfr.pop_dist(self.CFR.x, self.CFR.t),
                                     feed_dict={self.CFR.x: x_batch, self.CFR.t: t_batch})
        return m_statistics

    def run_h_rep(self, d_x_, do_in, do_out):
        return self.sess.run([self.CFR.h_rep],
                             feed_dict={self.CFR.x: d_x_, self.CFR.do_in: do_in, self.CFR.do_out: do_out})

    def run_weights(self):
        return self.sess.run(self.CFR.weights_in[0])

    def run_weights_pred(self):
        return self.sess.run(self.CFR.weights_pred)


class Train:
    def __init__(self, CFR, sess_runner, has_test, outdir, outform, outform_test, lossform, npzfile, npzfile_test,
                 repfile, repfile_test, logfile, ):
        self.repfile = repfile
        self.npzfile_test = npzfile_test
        self.npzfile = npzfile
        self.lossform = lossform
        self.outform_test = outform_test
        self.outform = outform
        self.outform = outform
        self.repfile_test = repfile_test
        self.outdir = outdir
        self.has_test = has_test
        self.CFR = CFR
        self.sess_runner = sess_runner
        self.train_step = self.create_train_step(CFR)
        self.sess_runner.initialize_global_variables()
        self.logfile = logfile

        if FLAGS.varsel:
            self.all_weights = None
            self.all_beta = None

    def create_train_step(self, CFR):

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(FLAGS.lrate, global_step, NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay,
                                        staircase=True)

        opt = self.choose_optimizer(lr)

        ''' Unused gradient clipping '''
        # gvs = opt.compute_gradients(CFR.tot_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
        # train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

        train_step = opt.minimize(CFR.tot_loss, global_step=global_step)
        return train_step

    def choose_optimizer(self, lr):

        if FLAGS.optimizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(lr)
        elif FLAGS.optimizer == 'GradientDescent':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif FLAGS.optimizer == 'Adam':
            opt = tf.train.AdamOptimizer(lr)
        else:
            opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)
        return opt

    def set_data(self, expr_dataset, expr_test_dataset, i_exp, I_valid):
        self.expr_dataset = expr_dataset
        self.expr_test_dataset = expr_test_dataset
        self.i_exp = i_exp
        self.I_valid = I_valid

        ''' Train/validation split '''
        n = self.expr_dataset['x'].shape[0]
        self.I_train = list(set(range(n)) - set(self.I_valid))
        self.n_train = len(self.I_train)

        """ Trains a CFR model on supplied data """
        ''' Compute treatment probability'''
        self.p_treated = np.mean(self.expr_dataset['t'][self.I_train, :])

        ''' Set up loss feed_dicts'''
        self.sess_runner.set_feed_dicts(self.expr_dataset, self.p_treated, self.I_train, self.I_valid)

    def train(self):

        ''' Set up for storing predictions '''
        preds_train = []
        preds_test = []

        ''' Compute losses '''
        losses = []
        obj_loss, f_error, imb_err = self.sess_runner.run_factual_losses()

        cf_error = np.nan
        if self.expr_dataset['HAVE_TRUTH']:
            cf_error = self.sess_runner.run_pred_loss()

        valid_obj = np.nan
        valid_imb = np.nan
        valid_f_error = np.nan

        if FLAGS.val_part > 0:
            self.sess_runner.run_factual_losses()
        else:
            self.dict_valid = dict(
                itertools.product(
                    [self.CFR.x, self.CFR.t, self.CFR.y_, self.CFR.do_in, self.CFR.do_out, self.CFR.r_alpha,
                     self.CFR.r_lambda, self.CFR.p_t],
                    np.array([])))

        losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

        objnan = False

        reps = []
        reps_test = []

        ''' Train for multiple iterations '''
        for i in range(FLAGS.iterations):
            objnan = self.train_once(self.I_train, i, losses, self.n_train, objnan, self.p_treated, preds_test,
                                     preds_train, reps, reps_test)

        return losses, preds_train, preds_test, reps, reps_test

    def should_compute_loss(self, i):
        ''' Compute loss every N iterations '''
        return i % FLAGS.output_delay == 0 or i == FLAGS.iterations - 1

    def train_once(self, I_train, i, losses, n_train, objnan, p_treated, preds_test, preds_train, reps, reps_test):
        t_batch, x_batch, y_batch = self.fetch_batch(I_train, n_train)

        if __DEBUG__:
            self.log_stats(t_batch, x_batch)

        if not objnan:
            self.gradient_step(p_treated, t_batch, x_batch, y_batch)

        ''' Project variable selection weights '''
        if FLAGS.varsel:  # TODO
            wip = simplex_project(self.sess_runner.run_weights(), 1)
            self.sess_runner.run_projection(wip)

        if self.should_compute_loss(i):
            objnan = self.compute_loss(i, losses, objnan, t_batch, x_batch, y_batch)

        if self.should_predict_in_M_iteration(i):
            self.predict(preds_test, preds_train, reps, reps_test)

        return objnan

    def predict(self, preds_test, preds_train, reps, reps_test):
        y_preds = self.run_yfact_and_ycfact(self.expr_dataset)
        preds_train.append(y_preds)
        if self.expr_test_dataset is not None:
            y_preds = self.run_yfact_and_ycfact(self.expr_test_dataset)
            preds_test.append(y_preds)
        if FLAGS.save_rep and self.i_exp == 1:
            reps_i = self.run_h_rep(self.expr_dataset['x'])
            reps.append(reps_i)

            if self.expr_test_dataset is not None:
                reps_test_i = self.run_h_rep(self.expr_test_dataset['x'])
                reps_test.append(reps_test_i)

    def compute_loss(self, i, losses, objnan, t_batch, x_batch, y_batch):
        obj_loss, f_error, imb_err = self.sess_runner.run_factual_losses()

        # TODO what the heck is this line?
        # rep = self.sess.run(self.CFR.h_rep_norm, feed_dict={self.CFR.x: self.D['x'], self.CFR.do_in: 1.0})
        rep = self.sess_runner.run_h_norm(self.expr_dataset['x'], 1.0)
        rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

        if self.expr_dataset['HAVE_TRUTH']:
            cf_error = self.sess_runner.run_pred_loss()
        else:
            cf_error = np.nan

        valid_obj = np.nan
        valid_imb = np.nan
        valid_f_error = np.nan

        if FLAGS.val_part > 0:
            valid_obj, valid_f_error, valid_imb = self.sess_runner.run_valid_losses()

        losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
        self.log_loss(cf_error, f_error, i, imb_err, obj_loss, t_batch, valid_f_error, valid_imb, valid_obj, x_batch,
                      y_batch)

        if np.isnan(obj_loss):
            log(self.logfile, 'Experiment %d: Objective is NaN. Skipping.' % self.i_exp)
            objnan = True

        return objnan

    def log_loss(self, cf_error, f_error, i, imb_err, obj_loss, t_batch, valid_f_error, valid_imb, valid_obj, x_batch,
                 y_batch):
        loss_str = '%d\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' % (
            i, obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

        if FLAGS.loss == 'log':
            acc = self.compute_accuracy(t_batch, x_batch, y_batch)
            loss_str += ',\tAcc: %.2f%%' % acc

        log(self.logfile, loss_str)

    def compute_accuracy(self, t_batch, x_batch, y_batch):
        y_pred = self.sess_runner.run_output(x_batch, t_batch, 1.0, 1.0)

        y_pred = 1.0 * (y_pred > 0.5)
        acc = 100 * (1 - np.mean(np.abs(y_batch - y_pred)))
        return acc

    def gradient_step(self, p_treated, t_batch, x_batch, y_batch):
        self.sess_runner.run_train_step(self.train_step, x_batch, t_batch, y_batch, p_treated)

    def log_stats(self, t_batch, x_batch):
        M = self.sess_runner(x_batch, t_batch)

        log(self.logfile,
            'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

    def fetch_batch(self, I_train, n_train):
        batch_indices = random.sample(range(0, n_train), FLAGS.batch_size)

        x_batch = self.expr_dataset['x'][I_train, :][batch_indices, :]
        t_batch = self.expr_dataset['t'][I_train, :][batch_indices]
        y_batch = self.expr_dataset['yf'][I_train, :][batch_indices]

        return t_batch, x_batch, y_batch

    def run_h_rep(self, d_x_):
        return self.sess_runner.run_h_rep(d_x_, 1, 0)

    def run_yfact_and_ycfact(self, dataset):
        t = dataset['t']
        x = dataset['x']
        y_pred_f = self.run_y_pred(x, t)
        y_pred_cf = self.run_y_pred(x, 1 - t)
        return np.concatenate([y_pred_f, y_pred_cf], axis=1)

    def run_y_pred(self, x, t):
        y_pred_f = self.sess_runner.run_output(x, t, 1.0, 1.0)
        return y_pred_f

    def should_predict_in_M_iteration(self, i):
        ''' Compute predictions every M iterations '''
        return (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i == FLAGS.iterations - 1

    def accumulate_weights(self):
        if self.i_exp == 1:
            self.all_weights = self.sess_runner.run_weights()
            self.all_beta = self.sess_runner.run_weights_pred()
        else:
            self.all_weights = np.dstack((self.all_weights, self.sess_runner.run_weights()))
            self.all_beta = np.dstack((self.all_beta, self.sess_runner.run_weights_pred()))

    def get_all_beta(self):
        return self.all_beta

    def get_all_weights(self):
        return self.all_weights

    def save_results(self, output_nodes):

        losses, preds_train, preds_test, reps, reps_test = self.train()

        output_nodes.collect_all_reps(losses, preds_test, preds_train)
        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(output_nodes.all_preds_train, 1, 3), 0, 2)
        if self.has_test:
            out_preds_test = np.swapaxes(np.swapaxes(output_nodes.all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(output_nodes.all_losses, 0, 2), 0, 1)
        ''' Store predictions '''
        log(self.logfile, 'Saving result to %s...\n' % self.outdir)

        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (self.outform, self.i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (self.outform_test, self.i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (self.lossform, self.i_exp), losses, delimiter=',')
        ''' Save results and predictions '''
        output_nodes.save_all_valid(self.I_valid)
        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            self.accumulate_weights()
            np.savez(self.npzfile, pred=out_preds_train, loss=out_losses, w=self.get_all_weights(),
                     beta=self.get_all_beta(),
                     val=output_nodes.get_all_valid())
        else:
            np.savez(self.npzfile, pred=out_preds_train, loss=out_losses, val=output_nodes.get_all_valid())
        if self.has_test:
            np.savez(self.npzfile_test, pred=out_preds_test)
        ''' Save representations '''
        if FLAGS.save_rep and self.i_exp == 1:
            np.savez(self.repfile, rep=reps)

            if self.has_test:
                np.savez(self.repfile_test, rep=reps_test)


class TrainRunner:
    def __init__(self, outdir):

        self.outdir = outdir
        self.npzfile = outdir + 'result'
        self.npzfile_test = outdir + 'result.test'
        self.repfile = outdir + 'reps'
        self.repfile_test = outdir + 'reps.test'
        self.outform = outdir + 'y_pred'
        self.outform_test = outdir + 'y_pred.test'
        self.lossform = outdir + 'loss'
        self.logfile = outdir + 'log.txt'
        self.n_experiments = self.calc_repetitions()

        self.set_random_seed()

        save_config(self.outdir + 'config.txt')

    def set_random_seed(self):
        random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(FLAGS.seed)

    def start_logs(self):
        f = open(self.logfile, 'w')
        f.close()

    def run(self):
        """ Runs an experiment and stores result in outdir """

        self.start_logs()
        dataform = FLAGS.datadir + FLAGS.dataform

        # ToDO
        has_test = False
        if not FLAGS.data_test == '':  # if test set supplied
            has_test = True
            dataform_test = FLAGS.datadir + FLAGS.data_test
        else:
            dataform_test = None

        log(self.logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha, FLAGS.p_lambda))

        ''' Load Data '''
        npz_input = False
        if dataform[-3:] == 'npz':
            npz_input = True
        if npz_input:
            datapath = dataform
            if has_test:
                datapath_test = dataform_test
        else:
            datapath = dataform % 1
            if has_test:
                datapath_test = dataform_test % 1

        log(self.logfile, 'Training data: ' + datapath)
        if has_test:
            log(self.logfile, 'Test data: ' + datapath_test)

        dataset = load_data_df(datapath)
        test_dataset = None
        if has_test:
            test_dataset = load_data_df(datapath_test)

        # log(self.logfile, 'Loaded data with shape [%d,%d]' % (dataset['n'], dataset['dim']))
        log(self.logfile, 'Loaded data with shape [%d,%d]' % (dataset.n, dataset.dim))

        CFR = self.creat_model(dataset)

        sess_runner = SessionRunner(CFR)

        output_nodes = Output()

        trainer = Train(CFR, sess_runner, has_test, self.outdir, self.outform,
                        self.outform_test, self.lossform, self.npzfile, self.npzfile_test, self.repfile,
                        self.repfile_test, self.logfile)

        ''' Run for all repeated experiments '''
        for i_exp in range(1, self.n_experiments + 1):

            if FLAGS.repetitions > 1:
                log(self.logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
            else:
                log(self.logfile, 'Training on experiment %d/%d...' % (i_exp, self.n_experiments))

            ''' Load Data (if multiple repetitions, reuse first set)'''

            expr_dataset, expr_test_dataset = self.prepare_data(dataset, test_dataset, dataform, dataform_test,
                                                                has_test, i_exp, npz_input)

            ''' Split into training and validation sets '''
            I_train, I_valid = validation_split(expr_dataset, FLAGS.val_part)

            ''' Run training loop '''
            trainer.set_data(expr_dataset, expr_test_dataset, i_exp, I_valid)
            trainer.save_results(output_nodes)

    def calc_repetitions(self):
        n_experiments = FLAGS.experiments
        if FLAGS.repetitions > 1:
            if FLAGS.experiments > 1:
                log(self.logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
                sys.exit(1)
            n_experiments = FLAGS.repetitions
        return n_experiments

    def creat_model(self, D):

        ''' Initialize input placeholders '''
        x = tf.placeholder("float", shape=[None, D.dim], name='x')  # Features
        t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatent
        y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

        ''' Parameter placeholders '''
        r_alpha = tf.placeholder("float", name='r_alpha')
        r_lambda = tf.placeholder("float", name='r_lambda')
        do_in = tf.placeholder("float", name='dropout_in')
        do_out = tf.placeholder("float", name='dropout_out')
        p = tf.placeholder("float", name='p_treated')

        ''' Define model graph '''
        log(self.logfile, 'Defining graph...\n')
        dims = [D.dim, FLAGS.dim_in, FLAGS.dim_out]
        CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

        return CFR

    def prepare_data(self, train_dataset, test_dataset, dataform, dataform_test, has_test, i_exp, npz_input):
        expr_dataset, expr_test_dataset = None, None
        if i_exp == 1 or FLAGS.experiments > 1:
            expr_test_dataset = None
            if npz_input:
                expr_dataset = train_dataset[i_exp - 1]

                if has_test:
                    expr_test_dataset = test_dataset[i_exp - 1]
            else:
                datapath = dataform % i_exp
                expr_dataset = load_data_df(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    expr_test_dataset = load_data_df(datapath_test)

        return expr_dataset, expr_test_dataset

    # def load_experiment_dataset(self, D, i_exp):
    #     D_exp = {}
    #     D_exp['x'] = D['x'][:, :, i_exp - 1]
    #     D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
    #     D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
    #     if D['HAVE_TRUTH']:
    #         D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
    #     else:
    #         D_exp['ycf'] = None
    #     return D_exp

    # def load_experiment_dataset(self, D, i_exp):
    #     result = {}
    #     result['x'] = D.df_x.xs(i_exp - 1).values
    #     result[t] = D.
    #
    #
    #     D_exp = {}
    #     D_exp['x'] = D['x'][:, :, i_exp - 1]
    #     D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
    #     D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
    #     if D['HAVE_TRUTH']:
    #         D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
    #     else:
    #         D_exp['ycf'] = None
    #     return D_exp


class Output:
    def __init__(self):
        ''' Set up for saving variables '''
        self.all_losses = []
        self.all_preds_train = []
        self.all_preds_test = []
        self.all_valid = []

    def collect_all_reps(self, losses, preds_test, preds_train):
        self.all_preds_train.append(preds_train)
        self.all_preds_test.append(preds_test)
        self.all_losses.append(losses)

    def save_all_valid(self, I_valid):
        self.all_valid.append(I_valid)

    def get_all_valid(self):
        return np.array(self.all_valid)


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir + '/results_' + timestamp + '/'
    os.mkdir(outdir)
    model = TrainRunner(outdir)

    try:
        model.run()
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()

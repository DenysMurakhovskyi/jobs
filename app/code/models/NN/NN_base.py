# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.compat import v1 as tf
tf.disable_v2_behavior()

from time import time
import abc
import numpy as np
from math import sqrt


def shuffle_in_unison_scary(lists):
    rng_state = np.random.get_state()
    for lst in lists:
        np.random.set_state(rng_state)
        np.random.shuffle(lst)


class NN_base(object):
    def __init__(self, batch_size=256, activation=tf.nn.relu, learning_rate=0.001, verbose=1, random_seed=2019, l2_reg=0.1):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.verbose = verbose
        self.activation = activation
        self.l2_reg = l2_reg
        self.weights = {}
        self.dropouts = {}


    def _init_session(self):
        #config = tf.ConfigProto(device_count={"gpu": 0})
        #config.gpu_options.allow_growth = True
        return tf.Session()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.loss = self._build_graph()
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            # init
            self.saver = tf.train.Saver(max_to_keep=1)
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord)
#            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
#            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            # number of params
            # print("#params: %d" % self._count_parameters())


    def _fully_connected(self, tensor, dim_input, dim_out, name, activation=None, bias=True):
        glorot = np.sqrt(2.0 / (dim_input + dim_out))
        if name + "_weights" not in self.weights: 
            self.weights[name + "_weights"] =  tf.Variable(np.random.normal(loc=0, scale=glorot, size=(dim_input, dim_out)),
                             dtype=np.float32, name=name + "_weights")
        y_deep = tf.matmul(tensor, self.weights[name + "_weights"])
        if bias:
            if name + "_bias" not in self.weights:
                self.weights[name + "_bias"] =  tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, dim_out)),
                                                            dtype=np.float32, name=name + "_bias")
            y_deep += self.weights[name + "_bias"]
        if activation != None:
            y_deep = activation(y_deep) 
        return y_deep, tf.keras.regularizers.l2(self.l2_reg)(self.weights[name + "_weights"])

    def _GCN(self, tensor, G, dim_input, layers, name, activation=None, bias=True):
        # F: N, maxlen, f, 
        # G: N, maxlen, maxlen
        if name not in self.dropouts:
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name=name+"_dropout")
        dropout = self.dropouts[name]
        y_deep = tf.nn.dropout(tensor, dropout[0])
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            y_deep, loss_now = self._fully_connected(y_deep, dim_input, layer, name=name+"_%d"%i, bias=bias, activation=activation)
            y_deep = tf.reshape(y_deep, [-1, self.maxlen_skill, layer])
            y_deep = tf.matmul(G, y_deep) + y_deep
            y_deep = tf.reshape(y_deep, [-1, layer])
            y_deep = tf.nn.dropout(y_deep, dropout[i + 1])
            loss += loss_now
            if (i + 1) % 3 == 0:
                y_deep += lst[i - 1]
            lst.append(y_deep)
            dim_input = layer
        return dropout, y_deep, loss

    def _GCN2(self, tensor, G, dim_input, layers, name, activation=None, bias=True):
        # F: N, maxlen, f, 
        # G: N, maxlen, maxlen
        if name not in self.dropouts:
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name=name+"_dropout")
        dropout = self.dropouts[name]
        y_deep = tf.nn.dropout(tensor, dropout[0])
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            E_d = tf.tile(tf.reshape(tf.reduce_sum(G, 2), [-1, self.maxlen_skill, 1]), [1, 1, layer])
            E_d = tf.where(tf.not_equal(E_d, 0), E_d, tf.ones_like(E_d))
            y_deep, loss_now = self._fully_connected(y_deep, dim_input, layer, name=name+"_%d"%i, bias=bias, activation=activation)
            y_deep = tf.reshape(y_deep, [-1, self.maxlen_skill, layer])
            y_deep = tf.matmul(G, y_deep) / E_d
            y_deep = tf.reshape(y_deep, [-1, layer])
            y_deep = tf.nn.dropout(y_deep, dropout[i + 1])
            loss += loss_now
            if (i + 1) % 3 == 0:
                y_deep += lst[i - 1]
            lst.append(y_deep)
            dim_input = layer
        return dropout, y_deep, loss


    def _mlp(self, tensor, dim_input, layers, name, activation=None, bias=True):
        if name not in self.dropouts: 
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name=name+"_dropout")
        dropout = self.dropouts[name]
        y_deep = tf.nn.dropout(tensor, dropout[0])
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            y_deep, loss_now = self._fully_connected(y_deep, dim_input, layer, name=name+"_%d"%i, bias=bias, activation=activation)
            y_deep = tf.nn.dropout(y_deep, dropout[i + 1])
            loss += loss_now
            if (i + 1) % 3 == 0:
                y_deep += lst[i - 1]
            lst.append(y_deep)
            dim_input = layer
        return dropout, y_deep, loss


    def fit(self, data_train, label_p, ind_p, data_valid=None, n_epoch=100):
        has_valid = data_valid != None
        valid_result = []
        for epoch in range(n_epoch):
            t1 = time()
            shuffle_in_unison_scary(data_train)
            predictions = self.run(data_train)
            pred_new = []
            import math
            for k in predictions:
                if math.isnan(k):
                    pred_new.append(0)
                else:
                    pred_new.append(k)
            predictions = pred_new
            train_eval = self.evaluate_metrics(predictions, data_train[label_p], data_train[ind_p])

            if has_valid:
                predictions = self.predict(data_valid)
                pred_new = []
                import math
                for k in predictions:
                    if math.isnan(k):
                        pred_new.append(0)
                    else:
                        pred_new.append(k)
                predictions = pred_new
                valid_eval = self.evaluate_metrics(predictions, data_valid[label_p], data_valid[ind_p])
                valid_result.append(valid_eval[0][1])

            if self.verbose > 0 and epoch % self.verbose == 0:
                print('[%d]' % epoch, end='')
                if has_valid:
                    self.print_result(train_eval, endch=' | ')
                    self.print_result(valid_eval, endch='')
                else:
                    self.print_result(train_eval, endch='')
                print('[%.1f s]' % (time() - t1))


    def _count_parameters(self):
        total_parameters = 0
        for name, variable in self.weights.items():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(name, variable_parameters)
            total_parameters += variable_parameters
        return total_parameters


    def save_model(self, save_path):
        self.saver.save(self.sess, save_path)


    def load_model(self, save_path):
        self.saver.restore(self.sess, save_path)


    def print_result(self, train_eval, endch="\n"):
        printstr = ""
        for i, name_val in enumerate(train_eval):
            if i != 0: printstr += ','
            printstr += '%s: %f' % name_val
        print(printstr, end=endch)


    def predict(self, data_test):
        predictions = self.run(data_test, train=False)
        return predictions


    def predict_and_evaluate(self, data_test, label_p, ind_p,):
        predictions = self.predict(data_test)
        pred_new = []
        import math
        for k in predictions:
            if math.isnan(k):
                pred_new.append(0)
            else:
                pred_new.append(k)
        predictions = pred_new
        test_eval = self.evaluate_metrics(predictions, data_test[label_p], data_test[ind_p])
        self.print_result(test_eval, endch='\n')
        return


    def evaluate_metrics(self, predictions, labels, tar_inds):
        low_predictions, high_predictions, low_labels, high_labels = [], [], [], []
        for pred, label, ind in zip(predictions, labels, tar_inds):
            if ind[0] == 1:
                low_predictions.append(pred)
                low_labels.append(label)
            else:
                high_predictions.append(pred)
                high_labels.append(label)
        # print(list(zip(low_predictions, low_labels))[:1000])
        low_r2 = r2_score(low_labels, low_predictions) if len(low_predictions) > 0 else 0
        high_r2 = r2_score(high_labels, high_predictions) if len(high_predictions) > 0 else 0
        low_mse = mean_squared_error(low_labels, low_predictions) if len(low_predictions) > 0 else 0
        low_mae = mean_absolute_error(low_labels, low_predictions) if len(low_predictions) > 0 else 0
        high_mse = mean_squared_error(high_labels, high_predictions) if len(high_predictions) > 0 else 0
        high_mae = mean_absolute_error(high_labels, high_predictions) if len(high_predictions) > 0 else 0
        # return [("low_mse", low_mse), ("low_mae", low_mae), ("low_r2", low_r2), ("high_mse", high_mse), ("high_mae", high_mae), ("high_r2", high_r2)]
        return [("low_rmse", sqrt(low_mse)), ("low_mae", low_mae), ("high_rmse", sqrt(high_mse)), ("high_mae", high_mae)]


    @abc.abstractmethod
    def _build_graph(self):
        '子类必须定义建图功能'
        pass


    @abc.abstractmethod #定义抽象方法，无需实现功能
    def get_dict(self, data, train=True):
        '子类必须定义get_dict功能'
        pass


    @abc.abstractmethod  # 定义抽象方法，无需实现功能
    def run(self, data, train=True):
        '子类必须定义run功能'
        pass


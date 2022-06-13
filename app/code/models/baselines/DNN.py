# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../../../../jobs/code/")
import tensorflow as tf
import pickle
import numpy as np
import os
from time import time
from CONFIG import *
from math import sqrt

def shuffle_in_unison_scary(lists):
    rng_state = np.random.get_state()
    for lst in lists:
        np.random.set_state(rng_state)
        np.random.shuffle(lst)

class DeepFM(object):
    def __init__(self, feat_dim, deep=[32, 32, 32], dropout_feed=[1.0, 1.0, 1.0], activation=tf.nn.relu, l2_reg=0.1,
                 learning_rate=0.001, verbose=1, random_seed=2019, batch_size=256):

        self.feat_dim = feat_dim
        self.deep = deep + [1]
        self.dropout_feed = dropout_feed + [1.0]
        self.batch_size = batch_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.random_seed = random_seed
        self.weights = {}
        self.dropouts = {}
        self._init_graph()

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

    def _mlp(self, tensor, dim_input, layers, name, activation=None, bias=True):
        if name not in self.dropouts:
            self.dropouts[name] = tf.compat.v1.placeholder(tf.float32, shape=[None], name=name+"_dropout")
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

    def _build_graph(self):

        # ------------------- input --------------------
        self.feat = tf.compat.v1.placeholder(tf.float32, shape=[None, self.feat_dim], name="feat")
        self.label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="label")  # N
        
        self.dropout_deep, self.pred, self.loss = self._mlp(self.feat, self.feat_dim, self.deep, name="mlp", activation=self.activation)
        # loss
        self.loss += tf.nn.l2_loss(self.label - self.pred)
        return self.loss

    def _init_session(self):
        #config = tf.ConfigProto(device_count={"gpu": 0})
        #config.gpu_options.allow_growth = True
        return tf.compat.v1.Session()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(self.random_seed)
            self.loss = self._build_graph()
            # optimizer
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)

            # init
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            init = tf.compat.v1.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)
            self.coord = tf.train.Coordinator()
            self.threads = tf.compat.v1.train.start_queue_runners(sess=self.sess,coord=self.coord)

            # number of params
            print("#params: %d" % self._count_parameters())

    def get_dict(self, data, train=True):
        n = len(data[1])
        feed_dict = {
                self.label: data[1],
                self.feat: data[0],
                self.dropout_deep: self.dropout_feed if train else [1] * len(self.dropout_feed),
                }
        return feed_dict

    def run(self, data, train=True):
        n_data = len(data[0])
        predictions = []
        for i in range(0, n_data, self.batch_size):
            data_batch = [dt[i:min(i+self.batch_size, n_data)] for dt in data]
            if train:
                preds, loss, _ = self.sess.run((self.pred, self.loss, self.optimizer), feed_dict=self.get_dict(data_batch, True))
            else:
                preds = self.sess.run(self.pred, feed_dict=self.get_dict(data_batch, False))
            predictions.extend(preds)
        return predictions 


    def fit(self, data_train, data_valid=None, n_epoch=100):
        has_valid = data_valid != None
        for epoch in range(n_epoch):
            t1 = time()
            shuffle_in_unison_scary(data_train)
            predictions = self.run(data_train)
            train_eval = self.evaluate_metrics(predictions, data_train[1])

            if has_valid:
                predictions = self.predict(data_valid)
                valid_eval = self.evaluate_metrics(predictions, data_valid[1])

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
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        return total_parameters

    def print_result(self, train_eval, endch="\n"):
        printstr = ""
        for i, name_val in enumerate(train_eval):
            if i != 0: printstr += ','
            printstr += '%s: %f' % name_val
        print(printstr, end=endch)


    def predict(self, data_test):
        predictions = self.run(data_test, train=False)
        return predictions


    def predict_and_evaluate(self, data_test):
        predictions = self.predict(data_test)
        test_eval = self.evaluate_metrics(predictions, data_test[1])
        self.print_result(test_eval, endch='\n')
        return


    def evaluate_metrics(self, predictions, labels):
        r2 = r2_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        return [("rmse", sqrt(mse)), ("mae", mae)]

def read_data(data_path, md=0):
    # ------------------- read information --------------
    with open(data_path, 'rb') as f:
        dct = pickle.load(f, encoding='utf-8')
    # dims
    n_level = dct['n_level']
    n_comp = dct['n_comp']
    n_city = dct['n_city']
    n_tempt = dct['n_tempt']
    n_workyear = dct['n_workyear']
    n_skill = dct['n_skill']
    n_industry = max([max(u[-1]) for u in dct['feat_comp']]) + 1

    # JD data
    time_ids = dct['time_index']
    salaries = dct['salary']
    ids = dct['id']
    skill_sets = dct['skill_set']
    comp_ids = dct['comp_id']
    city_ids = dct['city_id']
    work_years = dct['job_work_year']
    temptations = dct['job_temptation']

    # skill data
    data_skill_feat = dct['time_skill_feat']
    dim_skill = len(data_skill_feat[0][0])

    # company data
    feat_comp = dct['feat_comp']
    feat_time_city = dct['feat_time_city']
    feats, labels = [], []
    city_val_count = max([feat[2] for feat in feat_comp]) + 1
    for id, skill_set, comp_id, city_id, time_id, salary, work_year, tempt in zip(ids, skill_sets, comp_ids, city_ids, time_ids, salaries, work_years, temptations):
        # if time_id == 0:
        #     continue
        skill_feat2 = [0] * dim_skill
        skill_feat = [0] * n_skill
        for i, level_skill in enumerate(skill_set):
            skill_feat[level_skill[1]] = level_skill[0] + 1
            skill_feat2 = [sk_f + sk_n / len(skill_set) for sk_f, sk_n in zip(skill_feat2, data_skill_feat[0][level_skill[1]])]

        data_tempt = [0] * n_tempt
        for tempt_id in tempt:
            data_tempt[tempt_id] = 1
        industry_fields = feat_comp[comp_id][-1]
        industry_feat = [0] * n_industry
        for field in industry_fields:
            industry_feat[field] = 1
        company_city_feat = [0] * city_val_count
        company_city_feat[feat_comp[comp_id][2]] = 1

        feat = skill_feat + data_tempt + industry_feat + feat_comp[comp_id][:2] + feat_comp[comp_id][3:-1] + company_city_feat + feat_time_city[0][city_id] + [work_year, city_id, time_id] + skill_feat2
        if salary[md] != -1:
            feats.append(feat)
            labels.append(salary[md])
    return feats, labels

# 执行命令：python DNN.py {dataset_name} higher/lower
# 输出一次hold-out validation 结果


if __name__ == "__main__":
    # IT higher
    # dataset_name = "Designer"
    # salary_bound = "higher"
    dataset_name = 'IT'
    salary_bound = 'higher'
    md = 2 if salary_bound == 'higher' else 0
    
    data_path = "%s/%s/input_skillfeat.pkl" % (DATA_PATH, dataset_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data, label = read_data(data_path=data_path, md=md)
    feat_dim = len(data[0])
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2)#, random_state=0
    model = DeepFM(feat_dim, deep=[40, 40, 40, 32, 32, 32, 16, 16, 16], dropout_feed=[0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], activation=tf.nn.leaky_relu, l2_reg=0.01,
                   learning_rate=0.0001, verbose=20, random_seed=2019, batch_size=256)

    model.fit((train_X, train_y), (test_X, test_y), n_epoch=100) #300
    model.predict_and_evaluate((test_X, test_y))
    

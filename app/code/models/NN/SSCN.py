# -*- coding: utf-8 -*-
import pathlib
import sys

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import os
from app.code.models.NN.NN_base import NN_base
from app.code.CONFIG import *
from Tools import make_path


sys.path.append(str(pathlib.Path(__file__).parents[2].resolve()))


class SSCN(NN_base):
    def __init__(self, n_time, n_city, n_comp, n_industry, n_skill, n_level, n_tempt, n_workyear, dim_city, dim_skill, dim_comp,
                 comp_featsize, maxlen_skill, np_feat_city, np_feat_company_val, np_feat_company_id, np_feat_industry, np_skill_mat, np_skill_feat, np_G, time_reg=1,
                 fm_embsize=8, dropout_fm_feed=[1.0, 1.0], dim_skill_share=8, dim_skill_own=2, l2_reg=0.1,
                 deep_share=[32, 32], dropout_share_feed=[1.0,1.0,1.0],
                 deep_value=[32, 32], dropout_value_feed=[1.0, 1.0, 1.0],
                 deep_impact=[32, 32], dropout_impact_feed=[1.0, 1.0, 1.0],
                 gcn_layers=[32, 32], dropout_gcn_feed=[1.0, 1.0, 1.0],
                 hid_att=5,
                 activation=tf.nn.relu, learning_rate=0.001, verbose=1, random_seed=2019, batch_size=256):

        super(SSCN, self).__init__(batch_size=batch_size, activation=activation, learning_rate=learning_rate, l2_reg=l2_reg,
                                     verbose=verbose, random_seed=random_seed)

        # ------------------ data information ----------------
        self.n_time = n_time
        self.n_city = n_city
        self.n_comp = n_comp
        self.n_industry = n_industry
        self.n_skill = n_skill
        self.n_level = n_level
        self.n_workyear = n_workyear
        self.n_tempt = n_tempt
        self.dim_city = dim_city
        self.dim_comp = dim_comp
        self.comp_featsize = comp_featsize
        self.maxlen_skill = maxlen_skill
        self.dim_skill = dim_skill
        self.dim_skill_own = dim_skill_own

        # ------------------ pre input data -------------------
        self.np_feat_city = np_feat_city
        self.np_feat_company_val = np_feat_company_val
        self.np_feat_company_id = np_feat_company_id
        self.np_feat_industry = np_feat_industry
        self.np_skill_mat = np_skill_mat
        self.np_skill_feat = np_skill_feat
        self.np_G = np_G

        # ----------------- params ----------------------------
        # part1
        self.fm_embsize = fm_embsize
        self.dropout_fm_feed = dropout_fm_feed

        # part2
        self.dim_skill_sh = dim_skill_share
        self.deep_share = deep_share
        self.deep_value = deep_value
        self.deep_impact = deep_impact
        self.gcn_layers = gcn_layers
        self.dropout_share_feed = dropout_share_feed
        self.dropout_value_feed = dropout_value_feed
        self.dropout_in_feed = dropout_impact_feed
        self.dropout_out_feed = dropout_impact_feed
        self.dropout_gcn_feed = dropout_gcn_feed
        self.hid_att = hid_att
        self.time_reg = time_reg
        self.att = {}
        self._init_graph()


    def _graph_feature(self):
        G1, G2, G3, G4 = [], [], [], []

        # ------------------- discrete ------------------
        # workyear
        work_emb_tab = tf.Variable(tf.random.normal([self.n_workyear, self.fm_embsize], 0.0, 0.01), name="work_weights")
        self.weights['work_weights'] = work_emb_tab
        onehot_work = tf.one_hot(self.work_index, self.n_workyear) # N, n_workyear
        emb_work = tf.nn.embedding_lookup(params=work_emb_tab, ids=self.work_index) # N, fm_embsize
        emb_work = tf.reshape(emb_work, shape=[-1, 1, self.fm_embsize])
        G3.append((onehot_work, self.n_workyear))
        G4.append((emb_work, 1))

        # company
        feat_company_val = tf.constant(self.np_feat_company_val, name="feat_company_val")
        feat_company_id = tf.constant(self.np_feat_company_id, name="feat_company_id")
        company_emb_tab = tf.Variable(tf.random.normal([self.comp_featsize, self.fm_embsize], 0.0, 0.01), name="company_embeddings")
        self.weights["company_embeddings"] = company_emb_tab
        company_emb_tab = tf.nn.embedding_lookup(params=company_emb_tab, ids=feat_company_id)  # N, dim_comp, fm_embsize
        feat_value = tf.reshape(feat_company_val, shape=[-1, self.dim_comp, 1])  # N, D_companyfeat, 1
        company_emb_tab = tf.multiply(company_emb_tab, feat_value)  # N, dim_comp, fm_embsize
        company_emb_tab_flat = tf.reshape(company_emb_tab, [self.n_comp, self.dim_comp * self.fm_embsize])
        emb_comp_flat = tf.nn.embedding_lookup(params=company_emb_tab_flat, ids=self.comp_index) # N, self.dim_comp * self.fm_embsize
        emb_comp = tf.reshape(emb_comp_flat, [-1, self.dim_comp, self.fm_embsize]) # N, self.dim_comp, self.fm_embsize
        onehot_comp = tf.reduce_sum(input_tensor=tf.one_hot(feat_company_id, self.comp_featsize), axis=1) # N, comp_featsize
        onehot_comp = tf.nn.embedding_lookup(params=onehot_comp, ids=self.comp_index) # N, comp_featsize
        G3.append((onehot_comp, self.comp_featsize))
        G4.append((emb_comp, self.dim_comp))

        # city
        emb_city_tab = tf.Variable(tf.random.normal([self.n_city, self.fm_embsize], 0.0, 0.01), name="city_weights")
        self.weights['city_weights'] = emb_city_tab
        emb_city = tf.nn.embedding_lookup(params=emb_city_tab, ids=self.city_index)
        onehot_city = tf.one_hot(self.city_index, self.n_city)
        emb_city = tf.reshape(emb_city, [-1, 1, self.fm_embsize]) 
        G3.append((onehot_city, self.n_city))
        G4.append((emb_city, 1))

        # -------------------- continuous ------------------
        # industry
        feat_industry = tf.constant(self.np_feat_industry, dtype=tf.float32, name="feat_industry") # N, n_industry
        w_industry = tf.Variable(tf.random.normal([self.n_industry, self.fm_embsize], 0.0, 0.01), name="industry_embeddings")
        self.weights['industry_embeddings'] = w_industry
        ori_industry = tf.nn.embedding_lookup(params=feat_industry, ids=self.comp_index)
        pro_industry = tf.matmul(ori_industry, w_industry)  # [N, n_industry] x [n_industry, fm_embsize] = [N, fm_embsize]
        pro_industry = tf.reshape(pro_industry, [-1, 1, self.fm_embsize])
        G1.append((ori_industry, self.n_industry))
        G2.append((pro_industry, 1))

        # city
        w_city = tf.Variable(tf.random.normal([self.dim_city, self.fm_embsize], 0.0, 0.01), name="city_embeddings")
        self.weights['city_embeddings'] = w_city
        feat_city = tf.constant(self.np_feat_city, shape=[self.n_time, self.n_city, self.dim_city], name="feat_city")
        feat_city_flat = tf.reshape(feat_city, [self.n_time * self.n_city, self.dim_city])
        time_city_index = self.time_index * self.n_city + self.city_index

        ori_city = tf.nn.embedding_lookup(params=feat_city_flat, ids=time_city_index)
        pro_city = tf.matmul(tf.cast(ori_city, np.float32), w_city)
        pro_city = tf.reshape(pro_city, [-1, 1, self.fm_embsize])
        G1.append((ori_city, self.dim_city))
        G2.append((pro_city, 1))

        # temptation
        w_tempt = tf.Variable(tf.random.normal([self.n_tempt, self.fm_embsize], 0.0, 0.01), name="tempt_embeddings")
        self.weights['tempt_embeddings'] = w_tempt
        ori_tempt = self.temptation_feat
        pro_tempt = tf.matmul(self.temptation_feat, w_tempt)  # N, fm_embsize
        pro_tempt = tf.reshape(pro_tempt, [-1, 1, self.fm_embsize])
        G1.append((ori_tempt, self.n_tempt))
        G2.append((pro_tempt, 1))

        feat_linear = tf.concat([tf.cast(f[0], np.float32) for f in G1] + [tf.cast(f[0], np.float32) for f in G3], axis=1)
        dim_linear = sum([f[1] for f in G1] + [f[1] for f in G3])

        feat_multi = tf.concat([f[0] for f in G2] + [f[0] for f in G4], axis=1)
        dim_multi = sum([f[1] for f in G2] + [f[1] for f in G4])

        feat_deep = tf.concat([tf.cast(f[0], np.float32) for f in G1] + [tf.reshape(tf.cast(f[0], np.float32), [-1, f[1] * self.fm_embsize]) for f in G4], axis=1)
        dim_deep = sum([f[1] for f in G1] + [f[1] * self.fm_embsize for f in G4])
        return feat_linear, dim_linear, feat_multi, dim_multi, feat_deep, dim_deep

    def _MSVN(self, feat_linear, dim_linear, feat_multi, dim_multi, feat_deep, dim_deep):

        # -------------------- level -----------------------------------
        w_level = tf.Variable(tf.random.normal([self.n_level, self.fm_embsize], 0.0, 0.01), name="level_embedding")
        self.weights["level_embeddings"] = w_level
        emb_level = tf.nn.embedding_lookup(params=w_level, ids=self.level_index)  # N, maxlen, embsize
        onehot_level = tf.one_hot(self.level_index, self.n_level) # N, maxlen, n_level

        G_linear, G_deep, G_multi = [], [], []
        # ---------------------- Skill feature -------------------------
        # features, 用于linear和deep
        time_skill_index = tf.tile(tf.reshape(self.time_index, [-1, 1]), [1, self.maxlen_skill]) * self.n_skill + self.skill_index # N, maxlen_skill
        feat_skill = tf.constant(self.np_skill_feat, shape=[self.n_time, self.n_skill, self.dim_skill], name="feat_skill")
        feat_skill_flat = tf.reshape(feat_skill, [self.n_time * self.n_skill, self.dim_skill])
        ori_skillfeat = tf.nn.embedding_lookup(params=feat_skill_flat, ids=time_skill_index) # N, maxlen_skill, dim_skill
        G_deep.append((ori_skillfeat, self.dim_skill))
        G_linear.append((ori_skillfeat, self.dim_skill))

        skill_emb_tab = tf.Variable(tf.random.normal([1, self.dim_skill, self.fm_embsize], 0.0, 0.01), name="skill_weights")
        self.weights['skill_weights'] = skill_emb_tab
        feat_skill_flat = tf.reshape(feat_skill_flat, [-1, self.dim_skill, 1])  # N * maxlen_skill, dim_skill, 1
        feat_skill_flat = tf.tile(feat_skill_flat, [1, 1, self.fm_embsize])  # N * maxlen_skill, dim_skill, fm_embsize
        emb_skillfeat = tf.multiply(feat_skill_flat, tf.tile(skill_emb_tab, [self.n_time * self.n_skill, 1, 1]))  # N * maxlen_skill, dim_skill, fm_embsize
        emb_skillfeat = tf.nn.embedding_lookup(params=emb_skillfeat, ids=time_skill_index)
        emb_skillfeat = tf.reshape(emb_skillfeat, [-1, self.maxlen_skill, self.dim_skill, self.fm_embsize])
        G_multi.append((emb_skillfeat, self.dim_skill))

        # content feature
        graphfeat_skill = tf.constant(self.np_skill_mat, dtype=tf.float32,name="graphemb_skill")
        ori_graphskill = tf.nn.embedding_lookup(params=graphfeat_skill, ids=self.skill_index) # N, maxlen_skill, dim_skill
        w_skillgraph = tf.Variable(tf.random.normal([self.dim_skill_sh, self.fm_embsize], 0.0, 0.01), name="level_embedding")
        self.weights["project_skillgraph"] = w_skillgraph
        pro_graphskill = tf.reshape(ori_graphskill, [-1, self.dim_skill_sh])
        pro_graphskill = tf.matmul(pro_graphskill, self.weights['project_skillgraph']) #-1, embsize
        pro_graphskill = tf.reshape(pro_graphskill, [-1, self.maxlen_skill, 1, self.fm_embsize])
        G_multi.append((pro_graphskill, 1))

        # ----------------------- Skill embedding -----------------------
        w_skill = tf.Variable(tf.random.normal([self.n_time * self.n_skill, self.dim_skill_own], 0.0, 0.01), name="skill_embedding")
        self.weights["skill_embedding"] = w_skill
        w_map = tf.Variable(tf.random.normal([self.dim_skill_own, self.fm_embsize], 0.0, 0.01), name="skill_embedding_map")
        w_skill = tf.matmul(w_skill, w_map)
        emb_skill = tf.nn.embedding_lookup(params=w_skill, ids=time_skill_index)  # N, maxlen, fm_embsize
        G_deep.append((emb_skill, self.fm_embsize))
        G_linear.append((emb_skill, self.fm_embsize))
        G_multi.append((tf.reshape(emb_skill, [-1, self.maxlen_skill, 1, self.fm_embsize]), 1))

        # ----------------------- inputs ---------------------------------
        feat_linear = tf.tile(tf.reshape(feat_linear, [-1, 1, dim_linear]), [1, self.maxlen_skill, 1])
        feat_linear = tf.concat([feat_linear] + [f[0] for f in G_linear], 2)
        dim_linear += sum([f[1] for f in G_linear])

        feat_deep = tf.tile(tf.reshape(feat_deep, [-1, 1, dim_deep]), [1, self.maxlen_skill, 1])
        feat_deep = tf.concat([feat_deep] + [f[0] for f in G_deep], 2)
        dim_deep += sum([f[1] for f in G_deep])

        feat_multi = tf.tile(tf.reshape(feat_multi, [-1, 1, dim_multi, self.fm_embsize]), [1, self.maxlen_skill, 1, 1])
        feat_multi = tf.concat([feat_multi] + [f[0] for f in G_multi], 2)
        dim_multi += sum([f[1] for f in G_multi])

        # ---------------------- multiplicative ---------------------------
        feat_multi_flat = tf.reshape(feat_multi, [-1, dim_multi, self.fm_embsize])
        y_multi_flat = self.multiplicative(feat_multi_flat) # N * maxlen_skill, embsize

        # ---------------------- deep -------------------------------------
        flat_y_deep = tf.reshape(feat_deep, [-1, dim_deep])
        self.dropout_share, y_deep_share, loss = self._mlp(flat_y_deep, dim_deep, self.deep_share,
                                                      activation=self.activation, name="deep_share")

        # -------------------- value ----------------
        self.dropout_value, y_deep_val, loss_now = self._mlp(y_deep_share, self.deep_share[-1], self.deep_value,
                                                    activation=self.activation, name="deep_value")
        loss += loss_now
        feat_linear_flat = tf.reshape(feat_linear, shape=[-1, dim_linear])
        y_value = tf.concat((feat_linear_flat, y_multi_flat, y_deep_val), 1)
        y_value, loss_now = self._fully_connected(y_value, self.deep_value[-1] + self.fm_embsize + dim_linear, 2,
                                                  activation=self.activation, name="fc_value")
        loss += loss_now
        y_value = tf.nn.relu(y_value)
        y_value = tf.reshape(y_value, [-1, self.maxlen_skill, 2])
        y_temp = tf.concat((tf.zeros_like(y_value[:, :, 0]), y_value[:, :, 0]), axis=1)
        y_value += tf.reshape(y_temp, [-1, self.maxlen_skill, 2])

        return y_deep_share, y_value, loss

    def multiplicative(self, fm_embeddings):

        # -1, field, embsize
        # ---------- second order term ---------------
        # sum_square part
        summed_emb = tf.reduce_sum(input_tensor=fm_embeddings, axis=1)
        summed_emb_square = tf.square(summed_emb)

        # square_sum part
        squared_emb = tf.square(fm_embeddings)
        squared_sum_emb = tf.reduce_sum(input_tensor=squared_emb, axis=1)

        # second order
        y_second = 0.5 * tf.subtract(summed_emb_square, squared_sum_emb)
        # -1, embsize
        return y_second


    def _get_subgraph(self):
        self.graph = tf.constant(self.np_G, shape=[self.n_skill, self.n_skill], name="skill_graph", dtype=tf.float32)
        self.sample_id = tf.compat.v1.placeholder(tf.int32, shape=[None], name="sample_index")
        subgraph = tf.nn.embedding_lookup(params=self.graph, ids=self.skill_index) # N, maxlen_skill, n_skill
        subgraph = tf.transpose(a=subgraph, perm=[0, 2, 1]) # N * n_skill, maxlen_skill
        
        # N, maxlen 
        self.flat_skill_index = self.skill_index + tf.tile(tf.reshape(self.sample_id, [-1, 1]), [1, self.maxlen_skill]) * self.n_skill
        subgraph = tf.nn.embedding_lookup(params=tf.reshape(subgraph, [-1, self.maxlen_skill]), ids=self.flat_skill_index) # N, maxlen, maxlen
        subgraph = tf.transpose(a=subgraph, perm=[0, 2, 1]) # N, maxlen, maxlen
        has_mat = tf.tile(tf.reshape(self.skill_has, [-1, self.maxlen_skill, 1]), [1, 1, self.maxlen_skill])
        subgraph = tf.multiply(has_mat, subgraph)
        subgraph = tf.multiply(tf.transpose(a=has_mat, perm=[0, 2, 1]), subgraph)
        return subgraph


    def softmax_mask(self, o_att):
        o_att = tf.reshape(o_att, [-1, self.maxlen_skill])
        indmat = self.skill_has
        o_att = tf.compat.v1.where(tf.not_equal(indmat, 0), o_att, tf.zeros_like(o_att) - 10)
        o_max = tf.reduce_max(input_tensor=o_att, axis=1)
        o_max = tf.tile(tf.reshape(o_max, [-1, 1]), [1, self.maxlen_skill])
        o_att = tf.compat.v1.where(tf.not_equal(indmat, 0), tf.exp(o_att - o_max) + 1e-9, tf.zeros_like(o_att))
        o_sum = tf.reduce_sum(input_tensor=o_att, axis=1) # N
        o_sum = tf.tile(tf.reshape(o_sum, [-1, 1]), [1, self.maxlen_skill]) # N, maxlen_skill
        o_att = o_att / o_sum # N, maxlen_skill
        return o_att


    def _ASDN(self, y_deep_share):
        self.subgraph = self._get_subgraph() # N, maxlen, maxlen

        # in: importance out:influence
        self.dropout_in, flat_deep_in, loss = self._mlp(y_deep_share, self.deep_share[-1], self.deep_impact,
                                                   activation=None, name="deep_in")
        self.dropout_out, flat_deep_out, loss_now = self._mlp(y_deep_share, self.deep_share[-1], self.deep_impact,
                                                         activation=None, name="deep_out")
        loss += loss_now

        # -------------- GCN --------------------
        self.dropout_gcn, flat_gcn, loss_now = self._GCN2(flat_deep_out, self.subgraph, self.deep_impact[-1], self.gcn_layers,
                                                         activation=self.activation, name="gcn")
        loss += loss_now

        # -------------- Attention ---------------
        deep_out = tf.reshape(flat_deep_out, [-1, self.maxlen_skill, self.deep_impact[-1]])
        tile_skill_has = tf.tile(tf.reshape(self.skill_has, [-1, self.maxlen_skill, 1]), [1, 1, self.deep_impact[-1]])
        deep_out = tf.multiply(deep_out, tile_skill_has)
        gobal = tf.tile(tf.reshape(tf.reduce_sum(input_tensor=deep_out, axis=1), [-1, 1, self.deep_impact[-1]]), [1, self.maxlen_skill, 1])
        gobal -= deep_out # N, maxlen_skill, F
        skill_count = tf.tile(tf.reshape(tf.reduce_sum(input_tensor=tile_skill_has, axis=1), [-1, 1, self.deep_impact[-1]]), [1, self.maxlen_skill, 1]) - 1 # N, F
        gobal = tf.reshape(gobal / skill_count, [-1, self.deep_impact[-1]])
        
        feat_att =  tf.concat((flat_gcn, gobal, flat_deep_in), 1)

        att_low, _ = self._fully_connected(feat_att, self.deep_impact[-1] * 2 + self.gcn_layers[-1], self.hid_att, activation=tf.nn.tanh, name="low_att_hid")
        att_low, _ = self._fully_connected(att_low, self.hid_att, 1, activation=tf.nn.relu, name="low_att")

        att_high, _ = self._fully_connected(feat_att, self.deep_impact[-1] * 2 + self.gcn_layers[-1], self.hid_att, activation=tf.nn.tanh, name="high_att_hid")
        att_high, _ = self._fully_connected(att_high, self.hid_att, 1, activation=tf.nn.relu, name="high_att")

        att_low = self.softmax_mask(att_low)
        att_high = self.softmax_mask(att_high)
        
        att_low = tf.reshape(att_low, [-1, self.maxlen_skill, 1])
        att_high = tf.reshape(att_high, [-1, self.maxlen_skill, 1])
        o_att = tf.concat((att_low, att_high), 2)

        return o_att, loss


    def _build_graph(self):

        # ------------------- input --------------------
        self.time_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name="time_index")
        self.comp_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name="comp_index")
        self.city_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name="city_index")
        self.work_index = tf.compat.v1.placeholder(tf.int32, shape=[None], name="work_index")
        self.temptation_feat = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_tempt], name="temptation_feat")
        self.skill_index = tf.compat.v1.placeholder(tf.int32, shape=[None, self.maxlen_skill], name="skill_index")  # N, maxlen_skill
        self.level_index = tf.compat.v1.placeholder(tf.int32, shape=[None, self.maxlen_skill], name="level_index")  # N, maxlen_skill
        self.label = tf.compat.v1.placeholder(tf.float32, shape=[None], name="label")  # N
        self.target_ind = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name="target_ind")  # N, 2
        self.skill_has = tf.compat.v1.placeholder(tf.float32, shape=[None, self.maxlen_skill], name="skill_has")  # N, maxlen_skill

        feat_linear, dim_linear, feat_multi, dim_multi, feat_deep, dim_deep = self._graph_feature()
        y_deep_share, self.val_skill, loss = self._MSVN(feat_linear, dim_linear, feat_multi, dim_multi, feat_deep, dim_deep)
        self.o_att, loss_skill = self._ASDN(y_deep_share)

        self.skill_value = tf.multiply(self.o_att, self.val_skill) # N, maxlen_skill, 2
        self.pred = tf.reduce_sum(input_tensor=self.skill_value, axis=1) # N, 2
        self.pred = tf.reduce_sum(input_tensor=tf.multiply(self.target_ind, self.pred), axis=1)
        one_ind = tf.reshape(self.target_ind[:, 0], [-1])

        # loss
        self.loss = tf.nn.l2_loss(self.label - self.pred + 1 * tf.multiply(self.label - self.pred, one_ind)) + loss_skill

        # time loss
        w_own_lst = tf.split(self.weights['skill_embedding'], self.n_time, 0)
        for i in range(1, len(w_own_lst)):
            self.loss += self.time_reg * 0.01 * tf.nn.l2_loss(w_own_lst[i] - w_own_lst[i - 1])

        return self.loss

    def get_dict(self, data, train=True):
        n = len(data[1])
        feed_dict = {
                self.sample_id: [i for i in range(n)],
                self.label: data[1],
                self.skill_has: data[2],
                self.skill_index: data[3],
                self.level_index: data[4],
                self.comp_index: data[5],
                self.city_index: data[6],
                self.time_index: data[7],
                self.work_index: data[8],
                self.temptation_feat: data[9],
                self.target_ind: data[-1],
                self.dropout_share: self.dropout_share_feed if train else [1] * len(self.dropout_share_feed),
                self.dropout_value: self.dropout_value_feed if train else [1] * len(self.dropout_value_feed),
                self.dropout_in: self.dropout_in_feed if train else [1] * len(self.dropout_in_feed),
                self.dropout_out: self.dropout_out_feed if train else [1] * len(self.dropout_out_feed),
                self.dropout_gcn: self.dropout_gcn_feed if train else [1] * len(self.dropout_gcn_feed),
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


    def get_skill_value(self, data_test): # 保留id, skill_value, skill_factor (低的那个)
        n_data = len(data_test[0])
        oattlst, skilllst, levellst, valuelst, highlst, idlst = [], [], [], [], [], []
        for i in range(0, n_data, self.batch_size):
            data_batch = [dt[i:min(i + self.batch_size, n_data)] for dt in data_test]
            skill_values, o_att = self.sess.run((self.val_skill, self.o_att), feed_dict=self.get_dict(data_batch, False))
            skill_has = data_batch[2]
            skill_index = data_batch[3] 
            level_index = data_batch[4]
            target_ind = data_batch[-1]
            ids = data_batch[0]
            for idnow, val_lst, has_lst, oatt_lst, ind_lst, l_ind_lst, t_ind in zip(ids, skill_values, skill_has, o_att, skill_index, level_index, target_ind):
                l = sum(has_lst)
                valuelst.extend([u[t_ind[1]] for u in val_lst[:l]])
                oattlst.extend([u[t_ind[1]] for u in oatt_lst[:l]])
                skilllst.extend(ind_lst[:l])
                levellst.extend(l_ind_lst[:l])
                highlst.extend([t_ind[1]] * l)
                idlst.extend([idnow] * l)
        df = pd.DataFrame(list(zip(idlst, skilllst, levellst, valuelst, oattlst, highlst)), columns=['jd_id', 'skill', 'level', 'value', 'oatt', 'high'])
        return df


    def get_skill_embeddings(self, data):
        n_data = len(data[0])
        data_batch = [dt[:min(self.batch_size, n_data)] for dt in data]
        skill_emb = self.sess.run(self.weights["skill_embedding"], feed_dict=self.get_dict(data_batch, False))
        return skill_emb


def read_data():

    # ------------------- read information --------------
    with open("%s/input_skillfeat.pkl"%DATA_PATH_NOW, 'rb') as f:
        dct = pickle.load(f, encoding='utf-8')

    # dims
    n_time = dct['n_time']
    n_skill = dct['n_skill']
    n_level = dct['n_level']
    n_comp = dct['n_comp']
    n_city = dct['n_city']
    n_tempt = dct['n_tempt']
    n_workyear = dct['n_workyear']
    maxlen_skill = dct['maxlen_skill']
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

    # company data
    feat_comp = dct['feat_comp']
    feat_time_city = dct['feat_time_city']

    # dictionaries
    comp_ind_dict = dct['comp_ind_dict']
    city_ind_dict = ['city_ind_dict']
    data_id, data_salary, data_skill_has, data_skill_id, data_level_id, data_comp_id, \
    data_city_id, data_time_id, data_target_id, data_work_year, data_temptation = [], [], [], [], [], [], [], [], [], [], []
    dim_skill = len(data_skill_feat[0][0])
    print(len(ids))
    # ------------------- generate jd data ------------------
    for id, skill_set, comp_id, city_id, time_id, salary, work_year, tempt in zip(ids, skill_sets, comp_ids, city_ids, time_ids, salaries, work_years, temptations):
        skill_index = [0] * maxlen_skill
        level_index = [0] * maxlen_skill
        skill_has = [0] * maxlen_skill
        for i, level_skill in enumerate(skill_set):
            skill_index[i] = level_skill[1]
            level_index[i] = level_skill[0]
            skill_has[i] = 1
        data_tempt = [0] * n_tempt
        for tempt_id in tempt:
            data_tempt[tempt_id] = 1
        for i in [0, 2]:
            if salary[i] == -1:# or i == 2:
                continue
            data_id.append(id)
            data_salary.append(salary[i])
            data_skill_has.append(skill_has)
            data_skill_id.append(skill_index)
            data_level_id.append(level_index)
            data_comp_id.append(comp_id)
            data_city_id.append(city_id)
            data_time_id.append(time_id)
            data_work_year.append(work_year)
            data_temptation.append(data_tempt)
            target_ind = [0, 0]
            target_ind[min(i, 1)] = 1
            data_target_id.append(target_ind)
      
    # company data
    data_comp_feat_id, data_comp_feat_val, data_comp_industry = [], [], []
    city_val_count = max([feat[2] for feat in feat_comp]) + 1
    dim_comp = 4
    comp_featsize = 0
    for feat in feat_comp:
        company_feat_value = feat[:2] + [1, 1]  # FM输入
        company_feat_index = [0, 1] + [feat[2] + 2, feat[3] + 2 + city_val_count]
        comp_featsize = max(comp_featsize, feat[3] + 2 + city_val_count + 1)
        industry_fields = feat[-1]
        industry_feat = [0] * n_industry
        for field in industry_fields:
            industry_feat[field] = 1
        data_comp_feat_id.append(company_feat_index)
        data_comp_feat_val.append(company_feat_value)
        data_comp_industry.append(industry_feat)

    # city data
    dim_city = len(feat_time_city[0][0])

    return (n_time, n_comp, n_industry, n_city, n_skill, n_level, n_tempt, n_workyear, dim_comp, dim_city, dim_skill, comp_featsize, maxlen_skill), \
           (data_comp_feat_id, data_comp_feat_val, data_comp_industry, feat_time_city, data_skill_feat), \
           (data_id, data_salary, data_skill_has, data_skill_id, data_level_id, data_comp_id, data_city_id,
            data_time_id, data_work_year, data_temptation, data_target_id), (comp_ind_dict, city_ind_dict)


def read_skill_level_list(n_skill, n_level, md):
    with open("%s/%s/skill_level_dict.pkl"%(DATA_PATH, md), 'rb') as f:
        filtered_skill_dct, skill_ind_dct, level_dct, level_ind_dct, tempt_dct, tempt_ind_dct = pickle.load(f)
    skill_lst = [[] for i in range(n_skill)]
    level_lst = [[] for i in range(n_level)]
    for skill, ind in skill_ind_dct.items():
        skill_lst[ind] = skill
    for level, ind in level_ind_dct.items():
        level_lst[ind] = level
    return skill_lst, level_lst


def read_w2v(md):
    df = pd.read_csv("%s/%s/w2v.csv"%(DATA_PATH, md)).drop("skill", axis=1)
    df = df.sort_values(by='skill_id', ascending=True)
    return df.drop("skill_id", axis=1).values


def read_graph(md):
    with open("%s/%s/skill_graph_one.pkl"%(DATA_PATH, md), 'rb') as f:
        G = pickle.load(f, encoding='bytes')
    return G


if __name__ == "__main__":
    dataset_name = 'IT'
    load_model = 'new'
    DATA_PATH_NOW = "%s/%s" % (DATA_PATH, dataset_name)
    SAVE_PATH_NOW = "%s/%s" % (SAVE_PATH, dataset_name)
    SAVE_PATH_OUT = "%s/%s/model/" % (OUT_PATH, dataset_name)
    make_path(SAVE_PATH_OUT)

    data_skill_mat = read_w2v(dataset_name)
    params, side_data, data, dicts = read_data()
    n_time, n_comp, n_industry, n_city, n_skill, n_level, n_tempt, n_workyear, dim_comp, dim_city, dim_skill, comp_featsize, maxlen_skill = params
    data_comp_feat_id, data_comp_feat_val, data_comp_industry, feat_time_city, data_skill_feat = side_data
    skill_list, level_list = read_skill_level_list(n_skill, n_level, dataset_name)
    data_G = read_graph(dataset_name)

    data_train_test = train_test_split(*data, test_size=0.2, random_state=0)
    data_train = [data_train_test[i] for i in range(0, len(data_train_test), 2)]
    data_test = [data_train_test[i] for i in range(1, len(data_train_test), 2)]

    model = SSCN(n_time, n_city, n_comp, n_industry, n_skill, n_level, n_tempt, n_workyear, dim_city, dim_skill, dim_comp, 
                 comp_featsize, maxlen_skill, feat_time_city, data_comp_feat_val, data_comp_feat_id, data_comp_industry, data_skill_mat, data_skill_feat, data_G, time_reg=1,
                 fm_embsize=16, dropout_fm_feed=[0.8, 0.8], dim_skill_share=25,dim_skill_own=6, 
                 deep_share=[64, 64, 64], dropout_share_feed=[0.6, 0.6, 0.8, 0.8], l2_reg=0.3,
                 deep_value=[16, 16, 16], dropout_value_feed=[0.8, 1.0, 1.0, 1.0],
                 deep_impact=[16, 16, 16], dropout_impact_feed=[1.0, 1.0, 1.0, 1.0],
                 gcn_layers=[16, 16], dropout_gcn_feed=[1.0, 1.0, 1.0],
                 activation=tf.nn.leaky_relu, learning_rate=0.001, verbose=1, random_seed=2019, batch_size=128)

    make_path('%s/model_save' % SAVE_PATH_OUT)
    if load_model != 'load':
        # fit and save a new model
        model.fit(data_train, 1, -1, data_test, n_epoch=150) #150
        model.save_model("%s/model_save/SSCN" % SAVE_PATH_OUT)
    else:
        # load the model
        model.load_model("%s/model_save/SSCN" % SAVE_PATH_NOW)

    # model evaluation
    model.predict_and_evaluate(data_test, 1, -1)

    # save intermediate variables
    make_path('%s/value_save' % SAVE_PATH_OUT)

    df_val_train = model.get_skill_value(data_train)
    df_val_train['skill_name'] = df_val_train['skill'].apply(lambda x: skill_list[x])
    df_val_train['level_name'] = df_val_train['level'].apply(lambda x: level_list[x])
    df_val_train.to_csv('%s/value_save/value_train.csv' % SAVE_PATH_OUT, index=False)

    df_val_test = model.get_skill_value(data_test)
    df_val_test['skill_name'] = df_val_test['skill'].apply(lambda x: skill_list[x])
    df_val_test['level_name'] = df_val_test['level'].apply(lambda x: level_list[x])
    df_val_test.to_csv('%s/value_save/value_test.csv' % SAVE_PATH_OUT, index=False)

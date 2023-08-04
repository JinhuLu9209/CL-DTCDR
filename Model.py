# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
from DataSet import DataSet
import sys
import os
import heapq
import math
import scipy.io as scio
import pickle
from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(dataName_A, dataName_B, K_size, topK=10):
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-dataName_A',
                        action='store',
                        dest='dataName_A',
                        default=dataName_A)
    parser.add_argument('-dataName_B',
                        action='store',
                        dest='dataName_B',
                        default=dataName_B)
    parser.add_argument('-negNum',
                        action='store',
                        dest='negNum',
                        default=7,
                        type=int)
    parser.add_argument('-userLayer',
                        action='store',
                        dest='userLayer',
                        default=[K_size, 2 * K_size, K_size])
    parser.add_argument('-itemLayer',
                        action='store',
                        dest='itemLayer',
                        default=[K_size, 2 * K_size, K_size])
    parser.add_argument('-KSize', action='store', dest='KSize', default=K_size)
    parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lambdad',
                        action='store',
                        dest='lambdad',
                        default=0.001)
    parser.add_argument('-lr', action='store', dest='lr', default=0.001)
    parser.add_argument('-ssl_temp', action='store', dest='ssl_temp', default=1)
    parser.add_argument('-ssl_reg_intra', action='store', dest='ssl_reg_intra', default=0.3)
    parser.add_argument('-ssl_reg_inter', action='store', dest='ssl_reg_inter', default=0.2)
    parser.add_argument('-maxEpochs',
                        action='store',
                        dest='maxEpochs',
                        default=50,
                        type=int)
    parser.add_argument('-batchSize',
                        action='store',
                        dest='batchSize',
                        default=512,
                        type=int)
    parser.add_argument('-earlyStop',
                        action='store',
                        dest='earlyStop',
                        default=5)
    parser.add_argument('-checkPoint',
                        action='store',
                        dest='checkPoint',
                        default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=topK)
    args = parser.parse_args()

    classifier = Model(args)

    classifier.run()


class Model:
    def __init__(self, args):
        self.dataName_A = args.dataName_A
        self.dataName_B = args.dataName_B
        self.KSize = args.KSize

        self.model_N2V_A = KeyedVectors.load_word2vec_format(
            "Node2vec_" + self.dataName_A + "_KSize_" + str(self.KSize) + ".emb", binary=False)
        self.model_N2V_B = KeyedVectors.load_word2vec_format(
            "Node2vec_" + self.dataName_B + "_KSize_" + str(self.KSize) + ".emb", binary=False)

        self.model_D2V_A = Doc2Vec.load("Doc2vec_" + self.dataName_A + "_VSize_" + str(self.KSize) + ".model")
        self.model_D2V_B = Doc2Vec.load("Doc2vec_" + self.dataName_B + "_VSize_" + str(self.KSize) + ".model")

        self.dataSet_A = DataSet(self.dataName_A)
        self.dataSet_B = DataSet(self.dataName_B)
        self.shape_A = self.dataSet_A.shape
        self.maxRate_A = self.dataSet_A.maxRate
        self.shape_B = self.dataSet_B.shape
        self.maxRate_B = self.dataSet_B.maxRate
        self.train_A = self.dataSet_A.train
        self.test_A = self.dataSet_A.test
        self.train_B = self.dataSet_B.train
        self.test_B = self.dataSet_B.test

        self.negNum = args.negNum
        self.testNeg_A = self.dataSet_A.getTestNeg(self.test_A, 99)
        self.testNeg_B = self.dataSet_B.getTestNeg(self.test_B, 99)

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.lambdad = args.lambdad
        self.ssl_reg_intra = args.ssl_reg_intra
        self.ssl_reg_inter = args.ssl_reg_inter
        self.ssl_temp = args.ssl_temp
        self.add_model()

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

    def add_placeholders(self):
        self.user_A = tf.placeholder(tf.int32)
        self.item_A = tf.placeholder(tf.int32)
        self.rate_A = tf.placeholder(tf.float32)
        self.drop_A = tf.placeholder(tf.float32)
        self.user_B = tf.placeholder(tf.int32)
        self.item_B = tf.placeholder(tf.int32)
        self.rate_B = tf.placeholder(tf.float32)
        self.drop_B = tf.placeholder(tf.float32)

    def add_embedding_matrix(self, model_N2V, shape):
        embedding = []
        emb_len = shape[0] + shape[1]
        index_list = model_N2V.index2word
        for index in range(emb_len):
            if str(index) in index_list:
                node_feature = model_N2V.get_vector(str(index))
                embedding.append(node_feature)
            else:
                node_feature = np.zeros([self.KSize], dtype=np.float32)
                embedding.append(node_feature)

        return np.array(embedding)

    def add_doc_embedding(self, model_D2V, shape):
        embedding = []
        emb_len = shape[0] + shape[1]
        for index in range(emb_len):
            embedding.append(model_D2V.docvecs[index])

        return np.array(embedding)

    def add_side_embedding(self, dataS, dataT, emb_S, shape):
        with open('./correlation_' + str(dataS) + '2' + str(dataT) + "_VSize_" + str(self.KSize) + '.pkl',
                  'rb') as f:
            simDict = pickle.load(f)
        user_side_emb = []
        for i in range(shape[0]):
            if simDict[i]:
                user_side_emb.append(emb_S[simDict[i][0]])
            else:
                user_side_emb.append(emb_S[i])

        return np.array(user_side_emb)

    def add_model(self):
        with tf.name_scope("inference"):
            self.u_embeddings_A, self.i_embeddings_A = self._create_mlp_embed('A', self.model_N2V_A, self.shape_A, self.user_A, self.item_A)
            self.u_embeddings_B, self.i_embeddings_B = self._create_mlp_embed('B', self.model_N2V_B, self.shape_B, self.user_B, self.item_B)

        with tf.name_scope("loss_A"):
            node_features_B = self.add_embedding_matrix(self.model_N2V_B, self.shape_B)
            user_side_emb_A = self.add_side_embedding(self.dataName_B, self.dataName_A, node_features_B, self.shape_A)

            self.cross_entropy_loss_A, self.y_A = self.create_cross_entropy_loss(self.u_embeddings_A, self.i_embeddings_A, self.rate_A, self.maxRate_A)
            self.ssl_loss_intra_A = self.calc_ssl_loss_intra(self.u_embeddings_A, self.i_embeddings_A, self.model_D2V_A, self.shape_A, self.user_A, self.item_A)
            self.ssl_loss_inter_A = self.calc_ssl_loss_inter(self.u_embeddings_A, node_features_B, user_side_emb_A, self.user_A, self.shape_B)
            self.loss_A = self.cross_entropy_loss_A + self.ssl_loss_intra_A + self.ssl_loss_inter_A

        with tf.name_scope("loss_B"):
            node_features_A = self.add_embedding_matrix(self.model_N2V_A, self.shape_A)
            user_side_emb_B = self.add_side_embedding(self.dataName_A, self.dataName_B, node_features_A, self.shape_B)

            self.cross_entropy_loss_B, self.y_B = self.create_cross_entropy_loss(self.u_embeddings_B, self.i_embeddings_B, self.rate_B, self.maxRate_B)
            self.ssl_loss_intra_B = self.calc_ssl_loss_intra(self.u_embeddings_B, self.i_embeddings_B, self.model_D2V_B, self.shape_B, self.user_B, self.item_B)
            self.ssl_loss_inter_B = self.calc_ssl_loss_inter(self.u_embeddings_B, node_features_A, user_side_emb_B, self.user_B, self.shape_A)
            self.loss_B = self.cross_entropy_loss_B + self.ssl_loss_intra_B + self.ssl_loss_inter_B

    def _create_mlp_embed(self, domain, model_N2V, shape, user, item):
        node_features = self.add_embedding_matrix(model_N2V, shape)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_input = tf.nn.embedding_lookup(node_features, user)
            user_W1 = init_variable([self.KSize, self.userLayer[0]], "user_W1_" + domain)
            user_out = tf.matmul(user_input, user_W1)
            # full-connected layers (MLP)
            for i in range(0, len(self.userLayer) - 1):
                W = init_variable([self.userLayer[i], self.userLayer[i + 1]], "user_W_" + domain + str(i + 2))
                b = init_variable([self.userLayer[i + 1]], "user_b_" + domain + str(i + 2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_input = tf.nn.embedding_lookup(node_features, shape[0] + item)
            item_W1 = init_variable([self.KSize, self.itemLayer[0]], "item_W1_" + domain)
            item_out = tf.matmul(item_input, item_W1)
            # full-connected layers (MLP)
            for i in range(0, len(self.itemLayer) - 1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i + 1]], "item_W_" + domain + str(i + 2))
                b = init_variable([self.itemLayer[i + 1]], "item_b_" + domain + str(i + 2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        return user_out, item_out

    def create_cross_entropy_loss(self, u_embeddings, i_embeddings, rate, maxRate):
        user_out = u_embeddings
        item_out = i_embeddings
        norm_user_output = tf.sqrt(
            tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(
            tf.reduce_sum(tf.square(item_out), axis=1))
        regularizer = tf.nn.l2_loss(user_out) + tf.nn.l2_loss(item_out)
        predict = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keepdims=False) / (
                norm_item_output * norm_user_output)
        predict = tf.maximum(1e-6, predict)
        # predict = tf.reduce_sum(tf.multiply(user_out_B, item_out_B), axis=1, keepdims=False)
        regRate = rate / maxRate
        losses = regRate * tf.log(predict) + (1 - regRate) * tf.log(1 - predict)
        loss = -tf.reduce_sum(losses)
        loss = loss + self.lambdad * regularizer

        return loss, predict

    def calc_ssl_loss_intra(self, u_embeddings, i_embeddings, model_D2V, shape, user, item):
        doc_feature = self.add_doc_embedding(model_D2V, shape)

        user_emb1 = u_embeddings
        user_emb2 = tf.nn.embedding_lookup(doc_feature, user)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        normalize_all_user_emb2 = tf.nn.l2_normalize(doc_feature[0:shape[0]], 1)
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        item_emb1 = i_embeddings
        item_emb2 = tf.nn.embedding_lookup(doc_feature, shape[0] + item)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        normalize_all_item_emb2 = tf.nn.l2_normalize(doc_feature[shape[0]:shape[0] + shape[1]],
                                                     1)
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)
        ssl_loss_item = -tf.reduce_sum(tf.log(pos_score_item / ttl_score_item))

        ssl_loss = self.ssl_reg_intra * (ssl_loss_user + ssl_loss_item)

        return ssl_loss

    def calc_ssl_loss_inter(self, u_embeddings, node_features, u_side_embeddings, user, shape):
        user_emb1 = u_embeddings
        user_emb2 = tf.nn.embedding_lookup(u_side_embeddings, user)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        normalize_all_user_emb2 = tf.nn.l2_normalize(node_features[0:shape[0]], 1)
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))
        ssl_loss = self.ssl_reg_inter * ssl_loss_user

        return ssl_loss

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step_A = optimizer.minimize(self.loss_A)
        self.train_step_B = optimizer.minimize(self.loss_B)

    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def run(self):
        best_hr_A = -1
        best_NDCG_A = -1
        best_epoch_A = -1
        allResults_A = []
        best_hr_B = -1
        best_NDCG_B = -1
        best_epoch_B = -1
        allResults_B = []
        print("Start Training!")

        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch, "=" * 20)
            self.run_epoch(self.sess)
            print('=' * 50)
            print("Start Evaluation!")
            topK = self.topK
            hr_A, NDCG_A, hr_B, NDCG_B = self.evaluate(self.sess, topK)
            allResults_A.append([epoch, topK, hr_A, NDCG_A])
            allResults_B.append([epoch, topK, hr_B, NDCG_B])
            print(
                "Epoch ", epoch,
                "Domain A: {} TopK: {} HR: {}, NDCG: {}".format(self.dataName_A, topK, hr_A, NDCG_B))
            print(
                "Epoch ", epoch,
                "Domain B: {} TopK: {} HR: {}, NDCG: {}".format(self.dataName_B, topK, hr_B, NDCG_B))
            if hr_A > best_hr_A:
                best_hr_A = hr_A
                best_epoch_A = epoch
            if NDCG_A > best_NDCG_A:
                best_NDCG_A = NDCG_A
            if hr_B > best_hr_B:
                best_hr_B = hr_B
                best_epoch_B = epoch
            if NDCG_B > best_NDCG_B:
                best_NDCG_B = NDCG_B
            print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)
        print("Domain A: Best hr: {}, NDCG: {}, At Epoch {}; Domain B: Best hr: {}, NDCG: {}, At Epoch {}".format(
            best_hr_A, best_NDCG_A, best_epoch_A, best_hr_B, best_NDCG_B, best_epoch_B))
        bestPerformance = [[best_hr_A, best_NDCG_A, best_epoch_A], [best_hr_B, best_NDCG_B, best_epoch_B]]
        matname = './result/CL_DTCDR_' + str(self.dataName_A) + '_' + str(
            self.dataName_B) + '_top@' + str(self.topK) + '_Result.mat'
        scio.savemat(
            matname, {
                'allResults_A': allResults_A,
                'allResults_B': allResults_B,
                'bestPerformance': bestPerformance
            })
        print("Training complete!")

    def run_epoch(self, sess, verbose=10):
        train_u_A, train_i_A, train_r_A = self.dataSet_A.getInstances(
            self.train_A, self.negNum)
        train_len_A = len(train_u_A)
        shuffled_idx_A = np.random.permutation(np.arange(train_len_A))
        train_u_A = train_u_A[shuffled_idx_A]
        train_i_A = train_i_A[shuffled_idx_A]
        train_r_A = train_r_A[shuffled_idx_A]

        train_u_B, train_i_B, train_r_B = self.dataSet_B.getInstances(
            self.train_B, self.negNum)
        train_len_B = len(train_u_B)
        shuffled_idx_B = np.random.permutation(np.arange(train_len_B))
        train_u_B = train_u_B[shuffled_idx_B]
        train_i_B = train_i_B[shuffled_idx_B]
        train_r_B = train_r_B[shuffled_idx_B]

        num_batches_A = len(train_u_A) // self.batchSize + 1
        num_batches_B = len(train_u_B) // self.batchSize + 1

        losses_A = []
        losses_B = []
        max_num_batches = max(num_batches_A, num_batches_B)
        for i in range(max_num_batches):
            min_idx = i * self.batchSize
            max_idx_A = np.min([train_len_A, (i + 1) * self.batchSize])
            max_idx_B = np.min([train_len_B, (i + 1) * self.batchSize])
            if min_idx < train_len_A:
                train_u_batch_A = train_u_A[min_idx:max_idx_A]
                train_i_batch_A = train_i_A[min_idx:max_idx_A]
                train_r_batch_A = train_r_A[min_idx:max_idx_A]
                feed_dict_A = self.create_feed_dict(train_u_batch_A, train_i_batch_A, 'A', train_r_batch_A)
                _, tmp_loss_A, _y_A = sess.run([self.train_step_A, self.loss_A, self.y_A], feed_dict=feed_dict_A)
                losses_A.append(tmp_loss_A)
            if min_idx < train_len_B:
                train_u_batch_B = train_u_B[min_idx:max_idx_B]
                train_i_batch_B = train_i_B[min_idx:max_idx_B]
                train_r_batch_B = train_r_B[min_idx:max_idx_B]
                feed_dict_B = self.create_feed_dict(train_u_batch_B, train_i_batch_B, 'B', train_r_batch_B)
                _, tmp_loss_B, _y_B = sess.run([self.train_step_B, self.loss_B, self.y_B], feed_dict=feed_dict_B)
                losses_B.append(tmp_loss_B)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {};'.format(i, max_num_batches, np.mean(losses_A[-verbose:])))
                sys.stdout.write('\r{} / {} : loss = {}'.format(i, max_num_batches, np.mean(losses_B[-verbose:])))
                sys.stdout.flush()
        loss_A = np.mean(losses_A)
        loss_B = np.mean(losses_B)
        print("\nMean loss in this epoch is: Domain A={};Domain B={}".format(loss_A, loss_B))
        return loss_A, loss_B

    def create_feed_dict(self, u, i, dataset, r=None, drop=None):
        if dataset == 'A':
            return {
                self.user_A: u,
                self.item_A: i,
                self.rate_A: r,
                self.drop_A: drop,
                self.user_B: u,
                self.item_B: [],
                self.rate_B: [],
                self.drop_B: drop
            }
        else:
            return {
                self.user_B: u,
                self.item_B: i,
                self.rate_B: r,
                self.drop_B: drop,
                self.user_A: u,
                self.item_A: [],
                self.rate_A: [],
                self.drop_A: drop
            }

    def evaluate(self, sess, topK):
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0

        hr_A = []
        NDCG_A = []
        testUser_A = self.testNeg_A[0]
        testItem_A = self.testNeg_A[1]
        hr_B = []
        NDCG_B = []
        testUser_B = self.testNeg_B[0]
        testItem_B = self.testNeg_B[1]
        for i in range(len(testUser_A)):
            target = testItem_A[i][0]
            feed_dict_A = self.create_feed_dict(testUser_A[i], testItem_A[i], 'A')
            predict_A = sess.run(self.y_A, feed_dict=feed_dict_A)

            item_score_dict = {}

            for j in range(len(testItem_A[i])):
                item = testItem_A[i][j]
                item_score_dict[item] = predict_A[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_A.append(tmp_hr)
            NDCG_A.append(tmp_NDCG)
        for i in range(len(testUser_B)):
            target = testItem_B[i][0]
            feed_dict_B = self.create_feed_dict(testUser_B[i], testItem_B[i], 'B')
            predict_B = sess.run(self.y_B, feed_dict=feed_dict_B)

            item_score_dict = {}

            for j in range(len(testItem_B[i])):
                item = testItem_B[i][j]
                item_score_dict[item] = predict_B[j]

            ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)

            tmp_hr = getHitRatio(ranklist, target)
            tmp_NDCG = getNDCG(ranklist, target)
            hr_B.append(tmp_hr)
            NDCG_B.append(tmp_NDCG)
        return np.mean(hr_A), np.mean(NDCG_A), np.mean(hr_B), np.mean(NDCG_B)


if __name__ == '__main__':
    tasks = [['amazon_movie', 'amazon_book'], ['amazon_movie', 'amazon_cd']]
    topList = [10]
    for topK in topList:
        for [domain_A, domain_B] in tasks:
            main(domain_A, domain_B, 64, topK)

# -*- Encoding:UTF-8 -*-
import numpy as np
import pandas as pd


class DataSet(object):
    def __init__(self, fileName):
        self.fileName = fileName
        # self.user_pool = set()
        # self.item_pool = set()
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getDataDict(self.train)
        self.testDict = self.getDataDict(self.test)

    def getData(self, fileName):
        print("Loading %s data set..." % (fileName))
        filePath = './Data/' + fileName + '/ratings.csv'
        data = []
        ratings = pd.read_csv(filePath)
        u = ratings.user.max() + 1
        i = ratings.item.max() + 1
        self.maxRate = ratings.rate.max()
        for row in ratings.itertuples():
            user = int(row.user)
            item = int(row.item)
            rate = float(row.rate)
            time = float(row.time)
            # rate = float(1.0)
            # self.user_pool.add(user)
            # self.item_pool.add(item)
            data.append((user, item, rate, time))
        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))
        return data, [u, i]

    def getTrainTest(self):
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3]))
        train = []
        test = []
        for i in range(len(data) - 1):
            user = data[i][0]
            item = data[i][1]
            rate = data[i][2]
            if data[i][0] != data[i + 1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))

        test.append((data[-1][0], data[-1][1], data[-1][2]))
        return train, test

    def getDataDict(self, data):
        dataDict = {}
        for i in data:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            item = i[1]
            rating = i[2]
            train_matrix[user][item] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            neglist = set()
            neglist.add(i[1])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict or (i[0], j) in self.testDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            neglist = set()
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]

import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def get_error_rate(pred, Y):
    return sum(pred!=Y) / len(pred)


class RandomForest:
    def __init__(self, ntrees=10):
        self.ntrees = ntrees
        self.trees = []
        self.labels = set()
    
    def subsample(self, X, Y):
        X_sample = []
        Y_sample = []
        while len(X_sample) < len(X):
            index = random.randrange(0, len(X))
            X_sample.append(X[index])
            Y_sample.append(Y[index])
        return np.array(X_sample), np.array(Y_sample)
    
    def fit(self, X, y):
        self.trees = []
        # get labels
        self.labels = set(y)
        # fit every decision tree
        for i in range(self.ntrees):
            # get temp_X_train, temp_Y_train with bagging
            temp_X_train, temp_Y_train = self.subsample(X, y)
            #train it with decision tree
            self.trees.append(DecisionTreeClassifier(max_features='log2'))
            self.trees[i].fit(temp_X_train, temp_Y_train)
        # print(self.trees[0].predict(X[0].reshape(1,-1)))

    def predict(self, X):
        n_samples = len(X)
        # compute all predicate scores
        pred_scores = []
        for tree in self.trees:
            pred_scores.append(tree.predict(X))
        # initial votes
        votes = list()
        for i in range(n_samples):
            votes.append(dict(zip(self.labels, [0]*len(self.labels))))
        # start to vote
        for score in pred_scores:
            for i in range(len(score)):
                votes[i][score[i]] += 1
        #get final scores
        final_scores = []
        for vote in votes:
            max_key = max(vote, key=vote.get)
            final_scores.append(max_key)
        return np.array(final_scores)

if __name__ == '__main__':
    with open('adult_dataset/adult_train_feature.txt') as f:
        X_train = pd.read_table(f, sep=' ', header=None)
        X_train = X_train.values

    with open('adult_dataset/adult_train_label.txt') as f:
        Y_train = pd.read_table(f, sep=' ', header=None)
        Y_train = Y_train.values.ravel()
        for i in range(len(Y_train)):
            if Y_train[i] == 0:
                Y_train[i] = -1
    
    with open('adult_dataset/adult_test_feature.txt') as f:
        X_test = pd.read_table(f, sep=' ', header=None)
        X_test = X_test.values
    
    with open('adult_dataset/adult_test_label.txt') as f:
        Y_test = pd.read_table(f, sep=' ', header=None)
        Y_test = Y_test.values.ravel()
        for i in range(len(Y_test)):
            if Y_test[i] == 0:
                Y_test[i] = -1

    randomForest = RandomForest()
    randomForest.fit(X_train, Y_train)
    pred_train = randomForest.predict(X_train)
    pred_test = randomForest.predict(X_test)
    print('T = %d, AUC on test data: %f, error rate: %f' % (randomForest.ntrees, roc_auc_score(Y_test, pred_test), get_error_rate(pred_test, Y_test)))


    # standard_rf = RandomForestClassifier()
    # standard_rf.fit(X_train, Y_train)
    # pred_test = standard_rf.predict(X_test)
    # print(roc_auc_score(Y_test, pred_test))

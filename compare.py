#! /usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from BoostMain import AdaBoost
from RandomForestMain import RandomForest


def get_accuracy(X, Y):
    return sum(X == Y) / len(X)


def compare_by_cross_val(X, Y, k=5):
    """
    compare Adaboost with RandomForest by cross-validate AUC value
    """
    # number of base classifiers from [start, end]
    start, end = 1, 100
    # draw two figures--AUC and Accuracy
    x_axis = range(start, end + 1, 1)
    figure1_y1_axis = []
    figure1_y2_axis = []
    figure2_y1_axis = []
    figure2_y2_axis = []
    for num in x_axis:
        boost_auc = []
        forest_auc = []

        boost_accuracy = []
        forest_accuracy = []

        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(X):
            # get train set and test set
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            # compute adaboost AUC
            adaboost = AdaBoost(T=num)
            pred_test, pred_test_score = adaboost.adaboost(X_train, Y_train, X_test, Y_test)
            boost_auc.append(roc_auc_score(Y_test, pred_test_score))
            boost_accuracy.append(get_accuracy(Y_test, pred_test))
            # compute forest AUC
            forest = RandomForest(ntrees=num)
            forest.fit(X_train, Y_train)
            pred_test, pred_test_score = forest.predict(X_test)
            forest_auc.append(roc_auc_score(Y_test, pred_test_score))
            forest_accuracy.append(get_accuracy(Y_test, pred_test))
        boost_auc = sum(boost_auc) / k
        forest_auc = sum(forest_auc) / k
        boost_accuracy = sum(boost_accuracy) / k
        forest_accuracy = sum(forest_accuracy) / k
        print('T = %d' % num)
        print('boost AUC: %f, accuracy: %f' % (boost_auc, boost_accuracy))
        print('forest AUC: %f, accuracy: %f' % (forest_auc, forest_accuracy))
        figure1_y1_axis.append(boost_auc)
        figure1_y2_axis.append(forest_auc)
        figure2_y1_axis.append(boost_accuracy)
        figure2_y2_axis.append(forest_accuracy)

    plt.figure(1)
    plt.plot(x_axis, figure1_y1_axis, color='g')
    plt.plot(x_axis, figure1_y2_axis, color='orange')
    plt.xlabel('Number of base classifiers')
    plt.ylabel('AUC')
    plt.title('Compare AdaBoost and RandomForest')

    plt.figure(2)
    plt.plot(x_axis, figure2_y1_axis, color='g')
    plt.plot(x_axis, figure2_y2_axis, color='orange')
    plt.xlabel('Number of base classifiers')
    plt.ylabel('Accuracy')
    plt.title('Compare AdaBoost and RandomForest')

    plt.show()


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

    compare_by_cross_val(X_train, Y_train)

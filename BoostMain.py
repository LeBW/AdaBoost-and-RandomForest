from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class AdaBoost:
    def __init__(self, T=5):
        self.T = 5
    def adaboost(self, X_train, Y_train, X_test, Y_test):
        #init
        dec_tree = DecisionTreeClassifier()
        n_train, n_test = len(X_train), len(X_test)
        distribute_weight = np.ones(n_train) / n_train
        pred_train, pred_test = np.zeros(n_train), np.zeros(n_test)
        T = self.T

        for i in range(T):
            # fit a base classifier
            dec_tree.fit(X_train, Y_train, sample_weight=distribute_weight)
            temp_pred_train = dec_tree.predict(X_train)
            temp_pred_test = dec_tree.predict(X_test)
            # compute miss, loss, alpha
            miss = [int(x) for x in temp_pred_train != Y_train]
            loss = np.dot(distribute_weight, miss)
            if loss > 0.5:
                break
            alpha = 0.5 * np.log(1/loss - 1)
            # add to prediction
            pred_train += alpha * temp_pred_train
            pred_test += alpha * temp_pred_test
            # update distribution_weight
            params = [1 if x==1 else -1 for x in miss]
            distribute_weight = distribute_weight * [np.exp(alpha*x) for x in params]
            distribute_weight = distribute_weight * (1 / sum(distribute_weight))

        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

        auc_score = roc_auc_score(Y_test, pred_test)
        print(auc_score)
        



if __name__ == '__main__':
    with open('adult_dataset/adult_train_feature.txt') as f:
        X_train = pd.read_table(f, sep=' ', header=None)
        X_train = X_train.values

    with open('adult_dataset/adult_train_label.txt') as f:
        Y_train = pd.read_table(f, sep=' ', header=None)
        Y_train = Y_train.values.ravel()
    
    with open('adult_dataset/adult_test_feature.txt') as f:
        X_test = pd.read_table(f, sep=' ', header=None)
        X_test = X_test.values
    
    with open('adult_dataset/adult_test_label.txt') as f:
        Y_test = pd.read_table(f, sep=' ', header=None)
        Y_test = Y_test.values.ravel()
    
    ada_boost = AdaBoost()

    ada_boost.adaboost(X_train, Y_train, X_test, Y_test)

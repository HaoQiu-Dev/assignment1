"""
Logistic regression model
"""

import numpy as np
import math

def trans_y_into_onehot(y_train, label_num):
    res = np.eye(label_num)[np.array(y_train).reshape(-1)]
    res = res.reshape(list(y_train.shape).append(label_num)) 
    res[res == 0] = -1
    return res


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    # def sigmoid(self, z: np.ndarray) -> np.ndarray:
    def sigmoid(self, z: np.ndarray) :
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        
        return 1/(1 + np.exp(-z))
        
        # pass

    # def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) :
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        
        y_in_onehot = trans_y_into_onehot(y_train, 10)
        print(y_in_onehot.shape)
        print(np.dot(self.w, X_train.T).shape)
        print(self.w.shape)
        print(X_train.shape)
        
        for _ in range(self.epochs):
            sig_derive = self.sigmoid(-1 * y_in_onehot.T * np.dot(self.w, X_train.T))
            sumOver_xi = np.dot(sig_derive*y_in_onehot.T, X_train)
            self.w = self.w + self.lr * ((self.weight_decay * self.w ) + (1/N) *sumOver_xi)

        return self.w


    # def predict(self, X_test: np.ndarray) -> np.ndarray:
    def predict(self, X_test: np.ndarray) :
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        predicted = X_test.dot(self.w.T)
        maxIdex = np.argmax(predicted,axis=1)    
        return maxIdex 
        # pass

"""
Linear Regression model
"""

import numpy as np

def trans_y_into_onehot(y_train, label_num):
    res = np.eye(label_num)[np.array(y_train).reshape(-1)]
    res = res.reshape(list(y_train.shape).append(label_num)) 
    return res

class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    # def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) :
        """Train the classifier.
        Use the linear regression update rule as introduced in the Lecture.
        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        
        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        y_in_onehot = trans_y_into_onehot(y_train, 10)

        print(X_train.shape)
        print(y_train.shape)
        print(self.w.shape)
        print(self.weight_decay)
        
        for _ in range(self.epochs):
            sumOver_xi = (np.dot(self.w, X_train.T)*2 - y_in_onehot.T)@X_train
            self.w = self.w - self.lr *((self.weight_decay * self.w)+((1/N)*sumOver_xi))
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
        maxIndex = np.argmax(predicted, axis = 1)
        return maxIndex
        
        
        # pass
    
    
    


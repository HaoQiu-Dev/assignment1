"""
K Nearest Neighbours Model
"""
import numpy as np
import heapq
from collections import Counter

class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)
            
        # TODO: implement me
        # print(distance)
        
        output = []
        for i in range(len(distance)):
            idex = heapq.nsmallest(k_test, range(len(distance[i])), distance[i].take)
            # print("len index"+str(len(idex)))
            # print("len norm"+str(len(distance[0])))
            temp = [self._y_train[i] for i in idex]
            # print("len possible label"+str(len(temp)))
            label_single = Counter(temp).most_common(1)[0][0]
            output.append(label_single)
        # print(output)
        return np.array(output)
        # pass
        

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
        output = []
        for i in x_test:
            tempX = i - self._x_train
            temp_result = list(np.linalg.norm(tempX,axis= 1))
            # print(np.linalg.norm(tempX,axis= 1).shape)
            output.append(temp_result)
        # print(len(output))
        return np.array(output)
        # pass
    
    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        output = []
        for i in range(len(x_test)):
            tempNorm = []
            for j in range(len(self._x_train)):
                dist = np.linalg.norm(x_test[i]-self._x_train[j])
                tempNorm.append(dist)
            output.append(tempNorm)
        # print(output[0])
        return np.array(output)
        # pass

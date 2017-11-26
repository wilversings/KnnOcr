import numpy as np
from operator import itemgetter
from collections import defaultdict

class MatrixKnn():
    """description of class"""

    def __init__(self, train_matrices, threshold):
        
        self.train_matrices = train_matrices
        self.threshold = threshold

    @staticmethod
    def euclidian_squared(mat1, mat2):
        return np.power(mat1 - mat2, 2).sum()

    def get_thresholded_match(self, matrix):
        indexes = list(map(lambda en: (MatrixKnn.euclidian_squared(np.matrix(en[1]), matrix), en[0]), enumerate(self.train_matrices)))
        indexes.sort(key=lambda x : x[0])

        knn_hash = defaultdict(lambda: [])
        for i in range(7):
            knn_hash[(indexes[i][1] + 1) // 500].append(i)

        return max(list(knn_hash), key=lambda x: len(knn_hash[x])), indexes[:7], knn_hash

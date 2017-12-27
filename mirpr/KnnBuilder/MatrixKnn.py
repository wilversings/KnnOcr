import numpy as np
from KnnBuilder.KnnMatchResult import KnnMatchResult
from operator import itemgetter
from collections import defaultdict

class MatrixKnn():
    """description of class"""

    def __init__(self, train_matrices, threshold_map = defaultdict(lambda: -1)):
        
        self.train_matrices = train_matrices
        self.thresholds = threshold_map

    @staticmethod
    def euclidian_squared(mat1, mat2):
        return np.power(mat1 - mat2, 2).sum()

    def calibrate_thresholds():
        pass

    def get_thresholded_match(self, matrix):
        indexes_unsorted = list(map(lambda mat: MatrixKnn.euclidian_squared(np.matrix(mat), matrix), self.train_matrices))
        indexes = list(enumerate(indexes_unsorted))
        indexes.sort(key=lambda x : x[1])

        knn_hash = defaultdict(lambda: [])
        for i in range(7):
            knn_hash[(indexes[i][0] + 1) // 500].append(i)

        guessed_number = max(list(knn_hash), key=lambda x: len(knn_hash[x]))
        score = sum(indexes_unsorted[guessed_number * 500: (guessed_number + 1) * 500])

        return KnnMatchResult(guessed_number, score, self.thresholds[guessed_number] < score)

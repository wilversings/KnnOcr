import numpy as np
from KnnBuilder.KnnMatchResult import KnnMatchResult
from operator import itemgetter
from collections import defaultdict

default_treshold = {
    0: 1539198846,
    1: 988460674,
    2: 1643104477,
    3: 1302632161,
    4: 1553022526,
    5: 1810169042,
    6: 99999999999999, # not recognisable
    7: 99999999999999, # not recognisable
    8: 1787737414,
    9: 99999999999999 # not recognisable
}

class MatrixKnn:

    k = 3

    #In pixels.
    scale_precission =          4
    #In pixels.
    translation_precission =    4
    #In pixels.
    train_matrix_size =         20
    #In pixels.
    min_size =                  46

    def __init__(self, train_matrices, threshold_map = default_treshold):
        
        for mat in train_matrices:
            assert(mat.shape[0] == mat.shape[1] and mat.shape[0] == MatrixKnn.train_matrix_size)
        self.train_matrices = train_matrices

        self.thresholds = threshold_map

    @staticmethod
    def euclidian_squared(mat1, mat2):
        return np.power(mat1 - mat2, 2).sum()

    def calibrate_thresholds():
        pass

    def get_thresholded_match(self, matrix):
        assert(matrix.shape[0] == matrix.shape[1] and matrix.shape[0] == MatrixKnn.train_matrix_size)

        indexes_unsorted = list(map(lambda mat: MatrixKnn.euclidian_squared(np.matrix(mat), matrix), self.train_matrices))
        indexes = list(enumerate(indexes_unsorted))
        indexes.sort(key=lambda x : x[1])

        knn_hash = defaultdict(lambda: [])
        for i in range(MatrixKnn.k):
            knn_hash[(indexes[i][0] + 1) // 500].append(i)

        guessed_number = max(list(knn_hash), key=lambda x: len(knn_hash[x]))
        score = sum(indexes_unsorted[guessed_number * 500: (guessed_number + 1) * 500])

        return KnnMatchResult(guessed_number, score, score <= self.thresholds[guessed_number] * 5/4)

    @staticmethod
    def __steps(start,end,n):
        if n<2:
            raise Exception("behaviour not defined for n<2")
        step = (end - start) / float(n - 1)
        return [int(round(start + x * step)) for x in range(n)]

    @staticmethod
    def __scale(mat):
        assert(mat.shape[0] == mat.shape[1] and mat.shape[0] >= 20)
        lines = MatrixKnn.__steps(0, mat.shape[0] - 1, MatrixKnn.train_matrix_size)
        return mat[lines][:,lines]


    def get_all_matches(self, matrix):

        rows = matrix.shape[0]
        cols = matrix.shape[1]
        scale_max = min(rows, cols)

        assert(scale_max >= MatrixKnn.train_matrix_size)
        
        for scale in range(MatrixKnn.min_size, scale_max, MatrixKnn.scale_precission):
            for row in range(0, rows - scale, MatrixKnn.translation_precission):
                for col in range(0, cols - scale, MatrixKnn.translation_precission):

                    yield self.get_thresholded_match(
                        MatrixKnn.__scale(matrix[row : row + scale, col : col + scale])
                    )


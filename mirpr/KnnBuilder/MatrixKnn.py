import numpy as np
import multiprocessing
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
    train_matrix_size =         28
    #In pixels.
    min_size =                  35
    #In pixels.
    scale_difference =          8


    def __init__(self, train_matrices, threshold_map = default_treshold):
        
        for mat in train_matrices:
            assert(mat.shape[0] == mat.shape[1] and mat.shape[0] == MatrixKnn.train_matrix_size)

        self.train_matrices =   train_matrices
        self.thresholds =       threshold_map

    @staticmethod
    def euclidian_squared(mat1, mat2):
        return np.power(mat1 - mat2, 2).sum(dtype='uint32')

    def calibrate_thresholds():
        pass

    def get_thresholded_match(self, matrix) -> KnnMatchResult:
        assert(matrix.shape[0] == matrix.shape[1] and matrix.shape[0] == MatrixKnn.train_matrix_size)

        indexes_unsorted =  list(map(
                                lambda mat: MatrixKnn.euclidian_squared(np.matrix(mat), matrix), 
                                self.train_matrices
                            ))
        indexes =           list(enumerate(indexes_unsorted))

        indexes.sort(key=lambda x : x[1])

        knn_hash = defaultdict(lambda: [])
        for i in range(MatrixKnn.k):
            knn_hash[(indexes[i][0] + 1) // 6000].append(i)

        guessed_number = max(list(knn_hash), key=lambda x: len(knn_hash[x]))
        score = np.ma.sum(indexes_unsorted[guessed_number * 6000: (guessed_number + 1) * 6000], dtype='uint64')

        return KnnMatchResult(guessed_number, score, score <= self.thresholds[guessed_number] * 5/4)

    @staticmethod
    def __steps(start,end,n):
        assert (n >= 2)

        step = (end - start) / float(n - 1)
        return [int(round(start + x * step)) for x in range(n)]

    @staticmethod
    def __scale(mat):
        assert(mat.shape[0] == mat.shape[1] and mat.shape[0] >= 20)

        lines =     MatrixKnn.__steps(0, mat.shape[0] - 1, MatrixKnn.train_matrix_size)
        return      mat[lines][:,lines]

    def __find_first_match(self, matrix, rows, cols, scale_max):

        for scale in range(scale_max, MatrixKnn.min_size, -MatrixKnn.scale_precission):
            for row in range(0, rows - scale, MatrixKnn.translation_precission):
                for col in range(0, cols - scale, MatrixKnn.translation_precission):

                    match = self.get_thresholded_match(
                        MatrixKnn.__scale(matrix[row : row + scale, col : col + scale])
                    )
                    if match.is_valid:
                        match.set_coords(row, col, scale)
                        return scale, row, col, match

    def get_all_matches(self, matrix):

        rows =          matrix.shape[0]
        cols =          matrix.shape[1]
        scale_max =     min(rows, cols)

        assert(scale_max >= MatrixKnn.train_matrix_size)
        
        scale, row, col, match = self.__find_first_match(matrix, rows, cols, scale_max)

        row_jmp = None
        col_jmp = None

        for row in range(row, rows - scale, MatrixKnn.translation_precission):

            while col < cols - scale:

                match = self.get_thresholded_match(
                    MatrixKnn.__scale(matrix[row : row + scale, col : col + scale])
                )
                if match.is_valid:
                    match.set_coords(row, col, scale)
                    col_jmp = int(scale * 2/3)
                    yield match
                else:
                    col_jmp = None

                col += col_jmp or MatrixKnn.translation_precission
            col = 0
        row = 0

    def get_expression(self, matrix):

        return ''.join(map(lambda x: str(x.guessed_number), self.get_all_matches(matrix)))


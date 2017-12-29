from _collections import defaultdict


class TrainMatrices:

    
    def __init__(self, train_matrices):
        self.__train_matrices = train_matrices
        self.__digit_map = defaultdict(lambda: [])
        for mat in train_matrices:
            self.__digit_map[mat[1]] = mat[0]

    def get_by_digit(digit):
        return self.__digit_map[digit]

    def __iter__(self):
        return self.__train_matrices.__iter__()



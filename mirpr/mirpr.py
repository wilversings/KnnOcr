
import numpy as np
import cv2
from matplotlib import pyplot as plt
from functools import reduce
from MatrixKnn import MatrixKnn

img = cv2.imread('train/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

flatten = reduce(lambda a, b: a + b, cells)

knn = MatrixKnn(flatten, None)

test_im = np.ndarray.astype(cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2GRAY), 'int64')

number, indexes, knn_hash = knn.get_thresholded_match(test_im)

print("Guessed digit: " + str(number));
print("With a score of: " + "{:,}".format((
    sum(map(lambda x : MatrixKnn.euclidian_squared(test_im, x), flatten[number * 500: (number + 1) * 500]))
)))
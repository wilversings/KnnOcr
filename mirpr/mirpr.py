
import numpy as np
import cv2
from matplotlib import pyplot as plt
from functools import reduce
from KnnBuilder.MatrixKnn import MatrixKnn

img = cv2.imread('train/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

flat = reduce(lambda a, b: a + b, cells)

knn = MatrixKnn(flat)

test_im = np.ndarray.astype(cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2GRAY), 'int64')

res = knn.get_thresholded_match(test_im)

print("Guessed digit: " + str(res.guessed_number));
print("With a score of: " + "{:,}".format(
    res.score
))
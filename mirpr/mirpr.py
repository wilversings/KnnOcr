from multiprocessing.spawn import freeze_support

import numpy as np
import cv2
from matplotlib import pyplot as plt
from functools import reduce
from KnnBuilder.MatrixKnn import MatrixKnn
import sys

def parse():
    with open('train/mnist', 'rb') as mnist:
        mnist.seek(0x10)
        bytes = np.ndarray.astype(np.array(bytearray(mnist.read())), 'int16')

    with open('train/mnist_labels', 'rb') as labels:
        labels.seek(8)
        label = bytearray(labels.read())

    mnist = np.array_split(np.array_split(bytes, 60000 * 28), 60000)

    assert(mnist[-1].shape[0] == 28 and mnist[-1].shape[1] == 28)
    assert(len(mnist) == 60000 and len(label) == 60000)

    zipped = list(map(lambda sample, index: (sample, index), mnist, label))
    return zipped

if __name__ == "__main__":
    print ("\nParsing training data...")

    knn = MatrixKnn(parse())

    print ("Done. Continue to scan")

    while True:
        input()
        print("Loading...")

        test_im = np.ndarray.astype(cv2.cvtColor(cv2.imread('test.png'), cv2.COLOR_BGR2GRAY), 'int16')
    
        print(list(knn.get_all_matches(test_im)))

import mirpr
from KnnBuilder.MatrixKnn import MatrixKnn
import numpy as np
import sys

def parse_test():
    with open('test/mnist_test', 'rb') as mnist:
        mnist.seek(0x10)
        bytes = np.ndarray.astype(np.array(bytearray(mnist.read())), 'int16')

    with open('test/mnist_test_labels', 'rb') as labels:
        labels.seek(8)
        label = bytearray(labels.read())

    mnist = np.array_split(np.array_split(bytes, 10000 * 28), 10000)

    assert(len(mnist) == 10000 and len(label) == 10000)
    return list(map(lambda sample, index: (sample, index), mnist, label))

if __name__ == "__main__":
    print("\nParsing test data...")
    tests = parse_test()
    print("Done!")
    print("\nParsing training data...")
    knn = MatrixKnn(mirpr.parse())
    print("Done!\n")

    bingos = 0
    total = 10000
    iter = 0
    for test in tests:

        print("\r-------------------------Loading... {}%. Accuracy so far: {}/{} ({}%)-------------------------"
              .format((iter / total) * 100, bingos, total, (bingos / (iter + 1)) * 100), end='')
        sys.stdout.flush()
        iter += 1

        res = knn.get_thresholded_match(test[0])
        if res.guessed_number == test[1]:
            bingos += 1

    print("Accuracy: {}%".format((bingos / total) * 100))

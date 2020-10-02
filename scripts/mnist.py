# my default import modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def main():
    
    # load_ = True
    load_ = False

    if load_:
        # MNISTをダウンロード
        mnist = fetch_openml('mnist_784')
        # 保存先のディレクトリを作成
        os.makedirs('../dataset', exist_ok=True)
        # MNISTを保存
        np.savez(
            '../dataset/mnist.npz',
            data=mnist.data,
            target=mnist.target.astype(np.int64))
    else:
        mnist = np.load('../dataset/mnist.npz')
        print(mnist.files)
        print(mnist['data'])
        print(mnist['target'])


if __name__ == '__main__':
    main()


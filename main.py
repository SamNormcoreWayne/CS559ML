import numpy as np
import pandas as pd

from testing import Testing
from training import Training
from numpy.linalg import LinAlgError
from processing import PimaProcessing

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.linalg import eigh

def main():
    process = PimaProcessing()
    '''
    Problem 1
    '''
    process.open_file_and_store_pca()
    train_data, test_data = process.data_split()
    # print()
    round = 10
    process.precessing()
    train = Training(train_data, process.prior_one_length, process.prior_zero_length)
    test = Testing(test_data)
    train.do_training()
    prior_one = train.prior_one
    print("prior_one: {}".format(prior_one))
    prior_zero = train.prior_zero   
    for i in range(round):
        post_one, post_zero = test.get_ans(prior_one, prior_zero)
        prior_one = post_one
        prior_zero = post_zero
    mean, stdvar = test.get_accuracy()

    print("mean accuray: {}, stadard var: {}".format(mean, stdvar))

    '''
    Problem 3
    '''
    process = PimaProcessing()
    process.open_file_and_store_general()
    for i in range(10):
        train_data, test_data = process.data_split()

        lda = LDA(store_covariance=True)
        X_train, y_train = np.array(train_data.iloc[:, :-1]), np.transpose(np.array(train_data.iloc[:, -1:]))
        lda.fit(X_train, y_train)
        X_test, y_test = np.array(test_data.iloc[:, :-1]), np.transpose(np.array(test_data[:, -1:]))
        score = lda.score(X_test, y_test)
        print(f"round: {i}, accuracy: {score}")
        tmp = score
        if score >= tmp:
            opt = lda
    rows = X_train.shape
    S = opt.covariance_ * rows
    S = np.matrix(S)
    niu = (opt.means_[1] - opt.means_[0]).reshape(2, 1)
    Sb = np.dot(np.transpose(niu), niu)
    Sb = np.matrix(Sb)
    try:
        direction = np.dot(np.linalg.inv(S), np.transpose(niu))
    except LinAlgError:
        w, v = eigh(a=Sb,b=S, lower=True)
        del w
        print(f"ans: {v[0]}")
    else:
        print(f"ans: {direction}")

if __name__ == "__main__":
    main()

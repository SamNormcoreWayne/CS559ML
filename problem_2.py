import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.sparse.linalg import eigs
from scipy.linalg import eig, eigh

def main():
    d1 = [[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]]
    d2 = [[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]]
    #d1 = [[-2, 1], [-5, -4], [-3, 1], [0, -3], [-8, -1]]
    #d2 = [[2, 5], [1, 0], [5, -1], [-1, -3], [6, 1]]
    X = np.append(d1, d2, axis=0)
    Y = np.append(np.full((1, len(d1)), 1), np.zeros(len(d2)))
    lda = LDA(store_covariance=True)
    lda.fit(X, Y)
    print(f"X Shape: {X.shape}")
    d1_np = np.transpose(np.array(d1))
    #print(d1_np)
    #print(np.array(d1))
    d2_np = np.transpose(np.array(d2))
    # print(d1_np.shape)
    d1_mean = np.mean(d1_np, axis=1).reshape(1, 2)
    d2_mean = np.mean(d2_np, axis=1).reshape(1, 2)
    print(d1_mean)
    print(d2_mean)
    #print(d1_mean.reshape(2, 1))
    niu = d1_mean - d2_mean
    print(f"dot: {niu.shape}")
    d1_cov = np.cov(d1_np)
    d2_cov = np.cov(d2_np)
    print(f"cov: {d1_cov}, cov_2: {d2_cov}")
    S = np.add((len(d1) - 1) * d1_cov, (len(d2) - 1) * d2_cov)
    S = np.matrix(S)
    Sb = np.dot(np.transpose(niu), niu)
    Sb = np.matrix(Sb)
    print(f"ans: {np.dot(np.linalg.inv(S), np.transpose(niu))}")
    print(f"shape S: {S}, shape SB: {Sb}")
    w, v = eigh(a=Sb,b=S, lower=True)
    print(f"v: {v}, w: {w}")
    coef = lda.coef_
    print(f"coef shape: {coef}")
    print(f"mean shape: {lda.means_}")
    print(f"cov shape: {lda.covariance_ * (len(d1) + len(d2))}")

if __name__ == "__main__":
    main()

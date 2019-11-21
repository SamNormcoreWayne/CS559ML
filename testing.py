import numpy as np
from numpy import dot
from processing import PimaProcessing


class Testing:
    def __init__(self, data):
        self.data = data
        self.correct = 0
        self.wrong = 0
        self.accuracy = list()

    def gaussian(self, cov, mean, vec):
        # print("covshape: ", cov)
        # print("det: ", np.linalg.det(cov))
        inverse_cov = np.linalg.inv(cov)
        # print("inverse: ", inverse_cov)
        tmp = dot(dot(vec - mean, inverse_cov), np.transpose(vec - mean))
        # print("tmpshape: ", tmp)
        return np.exp(np.negative(tmp) // 2)

    def get_ans(self, prior_one, prior_zero):
        test_zero = PimaProcessing.get_zero_in_var(self.data)[:3]
        test_one = PimaProcessing.get_one_in_var(self.data)[:3]
        # print("test:one: ", test_one)
        test_zero = np.transpose(test_zero[:3])
        test_one = np.transpose(test_one[:3])
        cov_one = np.cov(np.transpose(test_one))
        cov_zero = np.cov(np.transpose(test_zero))
        mean_one = np.mean(test_one, axis=0)
        # print("meanone: ", mean_one)
        # print("var: ", np.var(test_one, axis=0))
        # print("covshape: ", cov_one.shape)
        # print("covshape: ", cov_zero.shape)
        mean_zero = np.mean(test_zero)
        test_data = np.transpose(self.data[:3])
        # print("test_zeroshape: ", test_zero.shape)
        # print("cov:", cov_one, cov_zero)

        for i in range(test_data.shape[0]):
            tmp_prior_one = prior_one
            tmp_prior_zero = prior_zero
            lklhood_one = self.gaussian(cov_one, mean_one, test_data[i])
            lklhood_zero = self.gaussian(cov_zero, mean_zero, test_data[i])

            # print("lklhood shape:", lklhood_one)
            post_one = lklhood_one * tmp_prior_one
            post_zero = lklhood_zero * tmp_prior_zero
            # print(tmp_prior_one)
            # print(post_one, post_zero)
            if (post_one > post_zero) and (self.data[i, 3] is 0):
                self.correct + 1
            elif (post_one < post_zero) and (self.data[i, 3] is 1):
                self.correct + 1
            else:
                self.wrong + 1

        print(self.correct, self.wrong)
        self.accuracy.append(self.correct // (self.correct + self.wrong))
        return post_one, post_zero

    def get_accuracy(self):
        array_accu = np.array(self.accuracy)
        return np.mean(array_accu), np.sqrt(np.var(array_accu))

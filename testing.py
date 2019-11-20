import numpy as np
from numpy import dot
from processing import PimaProcessing


class Testing:
    def __init__(self, data):
        self.data = data
        self.correct = 0
        self.wrong = 0
        self.accuracy = list()

    def gaussian(self, data):
        cov = np.cov(data)
        print("datashape: ", data.shape)
        print("covshape: ", cov.shape)
        inverse_cov = np.linalg.inv(cov)
        tmp = dot(dot(data, inverse_cov), np.transpose(data))
        print("tmpshape: ", tmp.shape)
        return np.exp(np.negative(tmp) // 2)

    def get_ans(self, prior_one, prior_zero):
        test_zero = PimaProcessing.get_zero_in_var(self.data)[:3]
        test_one = PimaProcessing.get_one_in_var(self.data)[:3]
        # print("test:one: ", test_one)
        self.lklhood_zero = self.gaussian(test_zero)
        self.lklhood_one = self.gaussian(test_one)
        print("lklhood: ", self.lklhood_one)
        lklhood_one = self.lklhood_one
        lklhood_zero = self.lklhood_zero

        for i in range(lklhood_one.size):
            tmp_prior_one = prior_one
            tmp_prior_zero = prior_zero
            
            post_one = lklhood_one[i] * tmp_prior_one
            post_zero = lklhood_zero[i] * tmp_prior_zero

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

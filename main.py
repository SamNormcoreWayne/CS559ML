import numpy as np
from testingMLE import Testing
from trainingMLE import Training
from processing import PimaProcessing

from sklearn.neighbors import KNeighborsClassifier

def main():
    '''
        Question 2:
    '''
    process = PimaProcessing()
    process.open_file_and_store()
    process.list_to_array()
    round = 10
    
    for i in range(round):
        train_data, test_data = process.data_split()
        process.precessing()
        train = Training(train_data, process.prior_one_length, process.prior_zero_length)
        test = Testing(test_data)
        train.do_training()
        prior_one = train.prior_one
        # print("prior_one: {}".format(prior_one))
        prior_zero = train.prior_zero
        post_one, post_zero = test.get_ans(prior_one, prior_zero)
        prior_one = post_one
        prior_zero = post_zero
    mean, stdvar = test.get_accuracy()

    print("Question 2: mean accuray: {}, stadard var: {}".format(mean, stdvar))

    '''
        Question 3:
    '''
    knn = list()
    train_data, test_data = process.data_split()
    k1 = KNeighborsClassifier(1)
    k1.fit(np.transpose(train_data[:3]), train_data[3])
    knn.append(k1.score(test_data[:3], test_data[3]))
    print("Question 3: knn: ", knn)



if __name__ == "__main__":
    main()

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
    for i in [1, 5 ,11]:
        k1 = KNeighborsClassifier(i)
        # print("Shape: ", np.transpose(train_data[:3]).shape, train_data[3].shape)
        k1.fit(np.transpose(train_data[:3]), train_data[3])
        knn.append(k1.score(np.transpose(test_data[:3]), test_data[3]))
    knn = np.array(knn)
    print("Question 3: mean accuracy: {}, standard var: {}".format(np.mean(knn), np.sqrt(np.var(knn))))

if __name__ == "__main__":
    main()

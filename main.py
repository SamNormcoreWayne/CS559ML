from testing import Testing
from training import Training
from processing import PimaProcessing

def main():
    process = PimaProcessing()
    process.open_file_and_store()
    process.list_to_array()
    train_data, test_data = process.data_split()
    # print()
    round = 10
    process.precessing()
    train = Training(train_data, process.prior_one_length, process.prior_zero_length)
    test = Testing(test_data)
    train.do_training()
    prior_one = train.prior_one
    # print("prior_one: {}".format(prior_one))
    prior_zero = train.prior_zero   
    for i in range(round):
        post_one, post_zero = test.get_ans(prior_one, prior_zero)
        prior_one = post_one
        prior_zero = post_zero
    mean, stdvar = test.get_accuracy()

    print("mean accuray: {}, stadard var: {}".format(mean, stdvar))

if __name__ == "__main__":
    main()

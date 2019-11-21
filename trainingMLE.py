class Training:
    mean_one = 0.0
    mean_zero = 0.0
    var_one = 0.0
    var_zero = 0.0
    def __init__(self, data, length_one, length_zero):
        self.length_one = length_one
        self.length_zero = length_zero
        self.data = data

    def do_training(self):
        data_size = self.length_one + self.length_zero
        # print(self.length_one)
        # print("size: ", data_size)
        self.prior_one = self.length_one / data_size
        # print("in class: ", self.prior_one)
        self.prior_zero = self.length_zero / data_size

class Training:
    mean_one = 0.0
    mean_zero = 0.0
    var_one = 0.0
    var_zero = 0.0
    def __init__(self, mean_one, mean_zero, var_one, var_zero, length_one, length_zero):
        Training.mean_one = mean_one
        Training.mean_zero = mean_zero
        Training.var_one = var_one
        Training.var_zero = var_zero
        self.length_one = length_one
        self.length_zero = length_zero
        self.prior_zero = 0.0
        self.prior_one = 0.0

    def do_training(self):
        data_size = self.length_one + self.length_zero
        self.prior_one = self.length_one // data_size
        self.prior_zero = self.length_zero // data_size

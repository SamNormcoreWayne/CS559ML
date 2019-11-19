class Training:
    def __init__(self, mean_one, mean_zero, var_one, var_zero, length_one, length_zero):
        self.mean_one = mean_one
        self.mean_zero = mean_zero
        self.var_one = var_one
        self.var_zero = var_zero
        self.length_one = length_one
        self.length_zero = length_zero

    def do_training(self):
        
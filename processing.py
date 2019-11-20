import csv
import os
import numpy as np
from pandas import DataFrame as df
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import cm


class PimaProcessing:
    COL_NUM: int = lambda: 9
    path: str = os.path.abspath()
    fp: str = lambda: os.path.join(PimaProcessing.path, "data", "pima-indians-diabetes-clean.csv")

    def __init__(self):
        self.pregnant_times = list()
        self.plasma_glucose = list()
        self.blood_pressure = list()
        self.triceps = list()
        self.serum_insulin = list()
        self.body_mass = list()
        self.diabetes = list()
        self.age = list()
        self.var = list()
        self.params = np.array(0)
        self.open_file_and_store()

    def open_file_and_store(self) -> None:
        with open(PimaProcessing.fp) as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                self.pregnant_times.append(row[0])
                self.blood_pressure.append(row[2])
                self.age.append(row[7])

    def list_to_array(self) -> None:
        self.pregnant_times = np.asarray(self.pregnant_times)
        self.blood_pressure = np.asarray(self.blood_pressure)
        self.age = np.asarray(self.age)
        self.var = np.asarray(self.var)
        self.params = np.array(self.pregnant_times, self.blood_pressure, self.age)
        self.standarilization()
        self.params = np.append()

    def standarilization(self) -> None:
        scaler = StandardScaler()
        scaler.fit(self.params)
        print("mean: ", scaler.mean_)
        scaler.transform(self.params)

    '''
    def visualization(self):
        feature_name = ["pregnant", "pressure", "age"]
    '''

    def data_split(self, seed : int = 0) -> tuple:
        if seed is 0:
            x_train, x_test, y_train, y_test = train_test_split(self.params, test_size=0.5)
        if seed is not 0:
            x_train, x_test, y_train, y_test = train_test_split(self.params, test_size=0.5, random_state=seed)
        return np.array(x_train), np.array(x_test)

    def precessing(self):
        tmp = self.params
        tmp_zero = PimaProcessing.get_zero_in_var(tmp)
        tmp_one = PimaProcessing.get_one_in_var(tmp)
        # split data into var == 0 and var == 1

        # Add instance variables here. Errors might ouccr
        self.mean_zero = np.mean(tmp_zero[:3])
        self.mean_one = np.mean(tmp_one[:3])
        self.var_zero = np.var(tmp_zero[:3])
        self.var_one = np.var(tmp_one[:3])

        # Errors might occur
        [rows, self.prior_zero_length] = tmp_zero.shape
        [rows, self.prior_one_length] = tmp_one.shape

    @staticmethod
    def get_zero_in_var(data : np.ndarray) -> np.ndarray :
        tmp = df(data)
        tmp = tmp.loc[:, ~(tmp.iloc[3, :] == 1)]
        return np.array(tmp)

    @staticmethod
    def get_one_in_var(data: np.ndarray) -> np.ndarray :
        tmp = df(data)
        tmp = tmp.loc[:, ~(tmp.iloc[3, :] == 0)]
        return np.array(tmp)

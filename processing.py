import csv
import os
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import cm


class PimaProcessing:
    COL_NUM: int = lambda: 9
    path: str = os.path.dirname(os.path.realpath(__file__))
    fp: str = os.path.join(path, "data", "pima-indians-diabetes-clean.csv")

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

    def open_file_and_store(self) -> None:
        with open(PimaProcessing.fp) as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                self.pregnant_times.append(int(row[0]))
                self.blood_pressure.append(int(row[2]))
                self.age.append(int(row[7]))
                self.var.append(int(row[8]))

    def open_file_and_store_general(self) -> None:
        with open(PimaProcessing.fp) as data_file:
            data = pd.read_csv(data_file, delimiter=',')
            self.params = data

    def open_file_and_store_pca(self) -> None:
        with open(PimaProcessing.fp) as data_file:
            data = pd.read_csv(data_file, delimiter=',')
            class_data = data.iloc[:, -1:]
            data = data.iloc[:, :-1]
            pca = PCA(n_components=3)
            pca.fit(data)
            self.params = pd.concat([pca.transform(data), class_data], axis=1)

    def list_to_array(self) -> None:
        self.pregnant_times = np.asarray(self.pregnant_times)
        self.blood_pressure = np.asarray(self.blood_pressure)
        self.age = np.asarray(self.age)
        self.var = np.asarray(self.var)
        self.params = np.array([self.pregnant_times, self.blood_pressure, self.age])
        # print(self.params)
        # print("shape: {}".format(self.params.shape))
        self.standarilization()
        # print("shape: {}".format(self.params.shape))
        # print(self.var.shape)
        self.params = np.append(self.params, [self.var], axis=0)
        # print("shape: {}".format(self.params.shape))

    def standarilization(self) -> None:
        scaler = StandardScaler()
        scaler.fit(self.params)
        # print ("mean: ", scaler.mean_)
        self.params = scaler.transform(self.params)
        # print("self: ", self.params)
        # print ("mean: ", scaler.mean_)
        # print("var: ", scaler.var_)

    '''
    def visualization(self):
        feature_name = ["pregnant", "pressure", "age"]
    '''

    def data_split(self, seed : int = 0) -> tuple:
        if seed is 0:
            x_train, x_test = train_test_split(np.transpose(self.params), test_size=0.5)
        if seed is not 0:
            x_train, x_test = train_test_split(np.transpose(self.params), test_size=0.5, random_state=seed)
        self.train_data = np.transpose(x_train)
        self.test_data = np.transpose(x_test)
        # print("train_data_shape: {}".format(self.train_data.shape))
        return self.train_data, self.test_data

    def precessing(self):
        tmp = self.train_data
        tmp_zero = PimaProcessing.get_zero_in_var(tmp)
        tmp_one = PimaProcessing.get_one_in_var(tmp)
        # split data into var == 0 and var == 1

        # Errors might occur
        [rows, self.prior_zero_length] = tmp_zero.shape
        [rows, self.prior_one_length] = tmp_one.shape

    @staticmethod
    def get_zero_in_var(data : np.ndarray) -> np.ndarray :
        tmp = df(data)
        # print("data: {}".format(data))
        tmp = tmp.loc[:, ~(tmp.iloc[3, :] == 1)]
        # print("delete one:", tmp)
        return np.array(tmp)

    @staticmethod
    def get_one_in_var(data: np.ndarray) -> np.ndarray :
        tmp = df(data)
        tmp = tmp.loc[:, ~(tmp.iloc[3, :] == 0)]
        # print(tmp)
        return np.array(tmp)

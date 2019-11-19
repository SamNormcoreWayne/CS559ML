import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
        self.params = np.array()
        self.open_file_and_store()

    def open_file_and_store(self):
        with open(fp) as data_file:
            reader = csv.reader(data_file)
            for row in reader:
                self.pregnant_times.append(row[0])
                self.blood_pressure.append(row[3])
                self.age.append(row[7])

    def list_to_array(self):
        self.pregnant_times = np.asarray(self.pregnant_times)
        self.blood_pressure = np.asarray(self.blood_pressure)
        self.age = np.asarray(self.age)
        self.var = np.asarray(self.var)
        self.params = np.array(self.pregnant_times, self.blood_pressure, self.age, self.var)

    def standarilization(self):
        scaler = StandardScaler()
        scaler.fit(self.params)
        print("mean: ", scaler._mean)
        scaler.
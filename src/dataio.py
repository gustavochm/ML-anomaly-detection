from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class tennessee_dataset():
    def __init__(self, train_path, test_path):
        
        self.data_train = pd.read_csv(train_path)
        self.data_train_np = self.data_train.values

        # Normalize and plot train data
        self.scaler = StandardScaler()

        self.Data_train_nolabel = self.scaler.fit_transform(self.data_train_np[:,0:-1])
        self.label_train = self.data_train_np[:,-1]
        self.label_train = self.label_train.reshape((len(self.label_train), 1))
        self.Data_train = np.concatenate((self.Data_train_nolabel, self.label_train), axis=1)
        self.Data_train_df = pd.DataFrame(self.Data_train)
        self.Data_train_df.columns = self.data_train.columns



        self.data_test = pd.read_csv(test_path)
        self.data_test_np = self.data_test.values
        # Normalize and test data
        self.Data_test_nolabel = self.scaler.transform(self.data_test_np[:,0:52])
        self.label_test = self.data_test_np[:,52]
        self.label_test = self.label_test.reshape((len(self.label_test), 1))
        self.Data_test = np.concatenate((self.Data_test_nolabel, self.label_test), axis=1)
        self.Data_test_df = pd.DataFrame(self.Data_test)
        self.Data_test_df.columns = self.data_test.columns

    def get_train_dataset(self):
        train_tensor = torch.tensor(self.Data_train_nolabel, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        
        return train_dataset
    
    def get_test_dataset(self):
        test_tensor = torch.tensor(self.Data_test_nolabel, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor)
        
        return test_dataset

    def plot_data(self):
        
        x_axis_train = np.linspace(0, len(self.data_train), len(self.data_train))
        plt.plot(x_axis_train, self.Data_train[:,0:2])
        plt.plot(x_axis_train, self.Data_train[:,2], color='paleturquoise')
        plt.title("Data_train")
        plt.legend(labels=["x1","x2","x3"])
        plt.show()


if __name__ == '__main__':
    train_path = 'https://raw.githubusercontent.com/iraola/te-orig-fortran/main/datasets/braatz_anomaly_detection/train/d00.csv'

    test_path = 'https://raw.githubusercontent.com/iraola/te-orig-fortran/main/datasets/braatz_anomaly_detection/test/d01.csv'

    ds = tennessee_dataset(train_path, test_path)

    print(1)

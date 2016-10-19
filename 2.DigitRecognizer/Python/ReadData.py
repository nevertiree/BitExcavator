import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import cross_val_score


def get_csv_data(csv_path):
    # csv_object = csv.reader(open(csv_path, 'r'), delimiter=',')
    # csv_list = list(csv_object)
    #return np.array(csv_list[1:])
    raw_data = pd.read_csv(csv_path, skiprows=0, sep=",")
    return np.array(raw_data)

# dataset path
training_data_path = "E:\\WorkSpace\\MyKaggle\\DigitRecognizer\\DataSet\\train.csv"
testing_data_path  = "E:\\WorkSpace\\MyKaggle\\DigitRecognizer\\DataSet\\test.csv"
sample_data_path   = "E:\\WorkSpace\\MyKaggle\\DigitRecognizer\\DataSet\\sample_submission.csv"

# print get_csv_data(testing_data_path)

def split_feature_target(training_data):
    feature = training_data[:, 1:]
    target = training_data[:, 0]
    return feature,target
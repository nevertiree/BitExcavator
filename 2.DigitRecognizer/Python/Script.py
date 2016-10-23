import numpy as np

from Preprocess import *
from Algorithm import *
import pandas as pd
import  matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets,svm,metrics

from  sklearn.model_selection import train_test_split

# dataset path
training_data_path = "E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\train.csv"
testing_data_path  = "E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\test.csv"
sample_data_path   = "E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\sample_submission.csv"

# get training data, testing data and sample result
training_data = get_csv_data(training_data_path)
#testing_data = get_csv_data(testing_data_path)
#sample_data = get_csv_data(sample_data_path)

# split the feature and target from training data
feature, target = split_feature_target(training_data)

print_pca_result_by_lossy(feature,"E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\train_pca.csv",0.1)

#print_pca_result_by_lossy(testing_data,"E:\\WorkSpace\\Kaggle\\2.DigitRecognizer\\DataSet\\test_pca.csv",0.1)

# Choose a Machine Learning Classifier
#classifier = RandomForestClassifier(n_estimators=10)

# train the algorithm and see what is going on.
#practice_algorithm(classifier,feature_train,feature_test,target_train,target_test)

# finish training and print out the result csv
#print_predict_result(classifier, feature, target, testing_data, sample_data)